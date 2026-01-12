# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import config
from .model_adapter import LayerAdapter, ModelAdapter
from .model_utils import get_layer0_inputs, get_signals
from .slicing_scheduler import ConfigSlicingScheduler, ConstSlicingScheduler, SlicingScheduler
from .utils import cleanup_memory, map_tensors


def rotate_attention_inputs(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in layer_adapter.get_attention_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_attention_inputs(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the WQ, WK and WV matrices of the self-attention layer.
    for W in layer_adapter.get_attention_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.in_features = new_embedding_dimension

    # Skip shortcut slicing if None (e.g., for global PCA where shortcuts are eliminated)
    if layer_adapter.layer.attn_shortcut_Q is not None:
        layer_adapter.layer.attn_shortcut_Q = nn.Parameter(layer_adapter.layer.attn_shortcut_Q[:new_embedding_dimension, :])


def rotate_attention_output(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer_adapter.get_attention_output()

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=config.device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def slice_attention_output(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice output matrix of the self-attention layer.
    W = layer_adapter.get_attention_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension


def rotate_mlp_input(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the MLP input weights.
    for W in layer_adapter.get_mlp_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_mlp_input(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the MLP input weights.
    for W in layer_adapter.get_mlp_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.in_features = new_embedding_dimension


def rotate_mlp_output(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the MLP output weights and bias.
    W = layer_adapter.get_mlp_output()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=config.device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def slice_mlp_output(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the MLP output weights and bias.
    W = layer_adapter.get_mlp_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension


def rotate_embeddings(model_adapter: ModelAdapter, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    for W in model_adapter.get_embeddings():
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    # Run GC and cleanup GPU memory
    cleanup_memory()


def slice_embeddings(model_adapter: ModelAdapter, new_embedding_dimensions: dict[int, int]) -> None:
    # Slice the embeddings.
    for i, W in enumerate(model_adapter.get_embeddings()):
        W.weight.data = W.weight.data[:, : new_embedding_dimensions[i]]
        W.embedding_dim = new_embedding_dimensions[i]


def rotate_head(model_adapter: ModelAdapter, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model_adapter.get_lm_head()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_head(model_adapter: ModelAdapter, new_embedding_dimension: int) -> None:
    # Slice the head.
    lm_head = model_adapter.get_lm_head()
    lm_head.weight.data = lm_head.weight.data[:, :new_embedding_dimension]
    lm_head.in_features = new_embedding_dimension


def rotate_and_slice(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
) -> None:
    """
    Rotate and slice a model, with interleaved slicing and PCA calculations
    """
    if model_adapter.parallel_blocks:
        rotate_and_slice_parallel(model_adapter, dataloader, slicing_scheduler, apply_mask, final_orientation)
    else:
        rotate_and_slice_sequential(model_adapter, dataloader, slicing_scheduler, apply_mask, final_orientation)


@torch.no_grad()
def rotate_and_slice_sequential(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
) -> None:
    """
    Rotate and slice the provided model, with interleaved slicing and PCA calculations.

    This method works for models where the MLP block is computed after the attention block.
    """
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype

    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    layers = model_adapter.get_layers()
    slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=False)

    # rotate and slice embeddings
    eig_val, Q = pca_calc(inps, ignore_masks)
    Q = Q.to(device=config.device)
    if final_orientation == 'random':
        R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_embedding_dimensions()[0])
        Q = Q @ R.to(Q.device)
    rotate_embeddings(model_adapter, Q)
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    logging.info("Rotate and slice layers")
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = nn.Parameter(Q.T.clone().to(dtype=dtype))

        # rotate and slice the attention inputs to match previous layer
        rotate_attention_inputs(layer_adapter, Q)
        slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))

        # get signal between attention and mlp, rotate and slice
        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(
                torch.matmul(inp.to(device=config.device), Q.to(dtype=dtype))[
                    :, :, : slicing_scheduler.get_attention_input_dimension(idx)
                ].cpu(),
                args[i],
            )

        mlp_ln_inputs, _ = get_signals(layer_adapter, args, kwargs)
        eig_val, Q = pca_calc(mlp_ln_inputs, ignore_masks)
        Q = Q.to(device=config.device, dtype=torch.float64)
        if final_orientation == 'random':
            R = random_orthogonal_upper_left(
                Q.shape[0], slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)
            )
            Q = Q @ R.to(Q.device)

        layer.attn_shortcut_Q = nn.Parameter(
            torch.matmul(
                layer.attn_shortcut_Q,
                Q.to(dtype=dtype)[:, : slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)],
            )
        )
        rotate_attention_output(layer_adapter, Q)
        slice_attention_output(
            layer_adapter, slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)
        )

        layer.mlp_shortcut_Q = nn.Parameter(
            Q.T.clone().to(dtype=dtype)[: slicing_scheduler.get_mlp_input_dimension(idx), :]
        )
        rotate_mlp_input(layer_adapter, Q)
        slice_mlp_input(layer_adapter, slicing_scheduler.get_mlp_input_dimension(idx))

        # Run GC and cleanup GPU memory
        cleanup_memory()

        # now compute the outputs of the current layer/inputs for the next layer
        # with slicing between Attention and mlp.
        _, inps = get_signals(layer_adapter, args, kwargs)
        eig_val, Q = pca_calc(inps, ignore_masks)
        if final_orientation == 'random':
            R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_mlp_output_dimension(idx))
            Q = Q @ R.to(Q.device)

        layer.mlp_shortcut_Q = nn.Parameter(torch.matmul(layer.mlp_shortcut_Q, Q.to(dtype=dtype)))

        # optionally slice the mlp/head connection in the last layer
        rotate_mlp_output(layer_adapter, Q)
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))
        layer.mlp_shortcut_Q = nn.Parameter(layer.mlp_shortcut_Q[:, : slicing_scheduler.get_mlp_output_dimension(idx)])

        layer.to('cpu')

        # Run GC and cleanup GPU memory
        cleanup_memory()

    # rotate and slice head
    rotate_head(model_adapter, Q)
    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())

    # update model's slicing config
    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()
    logging.info("Rotate and slice layers done")


@torch.no_grad()
def rotate_and_slice_parallel(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
) -> None:
    """
    Rotate and slice a model, with interleaved slicing and PCA calculations

    This version works for models where the MLP block and the attention block are computed in parallel.
    """
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype

    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    layers = model_adapter.get_layers()
    slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=True)

    # rotate and slice embeddings
    _, Q = pca_calc(inps, ignore_masks)
    Q = Q.to(device=config.device)
    if final_orientation == 'random':
        R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_embedding_dimensions()[0])
        Q = Q @ R.to(Q.device)
    rotate_embeddings(model_adapter, Q)
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    logging.info("Rotate and slice layers")
    layers = model_adapter.get_layers()
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = nn.Parameter(Q.T.clone().to(dtype=dtype))

        # rotate and slice the inputs to match previous layer (both attention and mlp)
        rotate_attention_inputs(layer_adapter, Q)
        rotate_mlp_input(layer_adapter, Q)
        slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))
        slice_mlp_input(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))

        # update the input signals to this layer, and re-run it
        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(
                torch.matmul(inp.to(device=config.device), Q.to(dtype=dtype))[
                    :, :, : slicing_scheduler.get_attention_input_dimension(idx)
                ].cpu(),
                args[i],
            )

        # the simpler equivalent of get_signals
        outputs = []
        layer = layer.to(config.device)
        for layer_args_batch, layer_kwargs_batch in zip(args, kwargs):
            layer_args_batch, layer_kwargs_batch = map_tensors(
                [layer_args_batch, layer_kwargs_batch], device=config.device
            )
            out = layer(*layer_args_batch, **layer_kwargs_batch)
            if isinstance(out, tuple):
                out = out[layer_adapter.hidden_states_output_position]
            out = out.cpu()
            outputs.append(out)

        inps = outputs
        _, Q = pca_calc(inps, ignore_masks)

        if final_orientation == 'random':
            R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_mlp_output_dimension(idx))
            Q = Q @ R.to(Q.device)

        # update shortcut matrix
        layer.attn_shortcut_Q = nn.Parameter(torch.matmul(layer.attn_shortcut_Q, Q.to(dtype=dtype)))

        # optionally slice the mlp/head connection in the last layer
        rotate_mlp_output(layer_adapter, Q)
        rotate_attention_output(layer_adapter, Q)
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))
        slice_attention_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))

        # slice the shortcut (there is only one, we use attn_shortcut buffer)
        layer.attn_shortcut_Q = nn.Parameter(
            layer.attn_shortcut_Q[:, : slicing_scheduler.get_mlp_output_dimension(idx)]
        )

        layer.to('cpu')

        # Run GC and cleanup GPU memory
        cleanup_memory()

    # rotate and slice head
    rotate_head(model_adapter, Q)
    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())

    # update model's slicing config
    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()
    logging.info("Rotate and slice layers done")


@torch.no_grad()
def rotate(model_adapter: ModelAdapter, dataloader: torch.utils.data.DataLoader[torch.Tensor]) -> None:
    """
    Rotate a model.
    TODO: Make this gpu memory efficient.
    """
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype  # Get the dtype of the model.

    # List of layers to rotate.
    layers = model_adapter.get_layers()

    # Get the input of the first layer norm and calculate the Q_1
    inps, args, kwargs = [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)

    _, Q_1 = pca_calc(inps)
    Q_1 = Q_1.to(device=config.device)

    # Rotate the embeddings.
    rotate_embeddings(model_adapter, Q_1)

    # Rotate the rest of the model.
    logging.info("Rotate layers")
    for layer_adapter in tqdm(layers, unit="layer", desc="Rotating"):
        layer = layer_adapter.layer
        # Extract the inputs and outputs of the second layernorm input and calculate the Q_3
        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(inp, args[i])
        mlp_ln_inputs, outs = get_signals(layer_adapter, args, kwargs)
        _, Q_3 = pca_calc(mlp_ln_inputs)
        Q_3 = Q_3.to(device=config.device)
        _, Q_5 = pca_calc(outs)
        Q_5 = Q_5.to(device=config.device)

        # Rotate the Q, K and V matrices of the self-attention layer.
        rotate_attention_inputs(layer_adapter, Q_1)

        # Set the shortcut rotation matrix of the self-attention layer.
        layer.attn_shortcut_Q = nn.Parameter(torch.matmul(Q_1.clone().T, Q_3.clone()).to(device="cpu", dtype=dtype))

        # Rotate the Attention output matrix
        rotate_attention_output(layer_adapter, Q_3)

        # Rotate the MLP input
        rotate_mlp_input(layer_adapter, Q_3)

        # Set the shortcut rotation matrix of the MLP.
        layer.mlp_shortcut_Q = nn.Parameter(torch.matmul(Q_3.clone().T, Q_5.clone()).to(device="cpu", dtype=dtype))

        # Rotate MLP output
        rotate_mlp_output(layer_adapter, Q_5)

        # Run GC and cleanup GPU memory
        cleanup_memory()

        inps = outs  # The inputs to the next layer are the outputs from this one!
        Q_1 = Q_5  # first rotation in the next layer is the last one in this...

    rotate_head(model_adapter, Q_5)
    logging.info("Rotate layers done")


def slice_rotated_model(model_adapter: ModelAdapter, slicing_scheduler: SlicingScheduler | None = None) -> None:
    """
    TODO: Make this gpu memory efficient.
    """
    model_adapter.model.eval()
    layers = model_adapter.get_layers()
    if not slicing_scheduler:
        if model_adapter.slicing_conf.const_dimension is not None:
            # backward compatibility for when no config is available
            slicing_scheduler = ConstSlicingScheduler(model_adapter.slicing_conf.const_dimension)
            slicing_scheduler.setup(
                hidden_size=model_adapter.hidden_size,
                layers_num=len(layers),
                parallel_blocks=model_adapter.parallel_blocks,
            )
        else:
            slicing_scheduler = ConfigSlicingScheduler(model_adapter.slicing_conf)

    # slice embeddings
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    # slice layers
    for i, layer_adapter in enumerate(layers):
        layer = layer_adapter.layer
        # slice attn weights 2nd dim, attn shortcut 1st dim
        slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(i))

        # slice mlp input 2nd dimension
        slice_mlp_input(layer_adapter, slicing_scheduler.get_mlp_input_dimension(i))

        # slice mlp shortcut 1st dimension
        # slice mlp shortcut
        if not model_adapter.parallel_blocks:
            layer.mlp_shortcut_Q = nn.Parameter(layer.mlp_shortcut_Q[: slicing_scheduler.get_mlp_input_dimension(i), :])

        # slice mlp weights 1st dimension
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(i))

        if model_adapter.parallel_blocks:  # parallel case
            layer.attn_shortcut_Q = nn.Parameter(
                layer.attn_shortcut_Q[:, : slicing_scheduler.get_attention_output_dimension(i, match_head_dim=True)]
            )
            slice_attention_output(
                layer_adapter, slicing_scheduler.get_attention_output_dimension(i, match_head_dim=True)
            )
        else:  # sequential case
            layer.attn_shortcut_Q = nn.Parameter(
                layer.attn_shortcut_Q[:, : slicing_scheduler.get_attention_output_dimension(i, match_head_dim=False)]
            )
            layer.mlp_shortcut_Q = nn.Parameter(
                layer.mlp_shortcut_Q[:, : slicing_scheduler.get_mlp_output_dimension(i)]
            )

            # slice attention weights 1st dimension
            slice_attention_output(
                layer_adapter, slicing_scheduler.get_attention_output_dimension(i, match_head_dim=False)
            )

    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())


def random_orthogonal_upper_left(total_dim, upper_block_dim):
    """
    Create a square matrix where the upper left block is a random orthogonal matrix, and the remainder is the identity.
    """
    A = np.random.rand(upper_block_dim, upper_block_dim)
    Q, _ = np.linalg.qr(A)
    R = np.eye(total_dim)
    R[:upper_block_dim, :upper_block_dim] = Q
    return torch.from_numpy(R)


@torch.no_grad()
def pca_calc(
    X: list[torch.Tensor], ignore_masks: list[torch.Tensor] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run PCA on a list of batched data. Returns the eigenvalues and eigenvectors.
    """
    # Run GC and cleanup GPU memory
    cleanup_memory()

    H = None
    for idx, X_batch in enumerate(X):
        if ignore_masks:
            X_batch[ignore_masks[idx] == 0] = 0

        X_batch = X_batch.double().to(device=config.device)
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)  # sum over the batch dimension.
        H = H_batch if H is None else H + H_batch

    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).to(device=config.device)
    H[diag, diag] = H[diag, diag] + damp
    X_eig = torch.linalg.eigh(H)
    del H
    index = torch.argsort(X_eig[0], descending=True)
    eig_val = X_eig[0][index]
    eigen_vec = X_eig[1][:, index]
    return eig_val, eigen_vec

#### Global PCA Implementation ####

@torch.no_grad()
def collect_global_covariance(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    apply_mask: bool = True,
) -> torch.Tensor:
    """
    Compute global covariance by aggregating layer 0 inputs and all layer outputs.
    """
    model_adapter.model.eval()
    
    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])
    
    H = None
    for idx, X_batch in enumerate(inps):
        X_batch_masked = X_batch.clone()
        if ignore_masks:
            X_batch_masked[ignore_masks[idx] == 0] = 0
        X_batch_masked = X_batch_masked.double().to(device=config.device)
        H_batch = torch.sum(X_batch_masked.mT @ X_batch_masked, dim=0)
        H = H_batch if H is None else H + H_batch
        del X_batch_masked, H_batch
    
    cleanup_memory()
    
    layers = model_adapter.get_layers()
    current_inps = inps
    
    for layer_idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Collecting global covariance")):
        layer = layer_adapter.layer.to(config.device)
        layer_outputs = []
        for i, inp in enumerate(current_inps):
            layer_args = layer_adapter.get_updated_args(inp, args[i])
            layer_args = map_tensors(layer_args, device=config.device)
            layer_kwargs = map_tensors(kwargs[i], device=config.device)
            
            out = layer(*layer_args, **layer_kwargs)
            if isinstance(out, tuple):
                out = out[layer_adapter.hidden_states_output_position]
            layer_outputs.append(out.cpu())
        for idx, X_batch in enumerate(layer_outputs):
            X_batch_masked = X_batch.clone()
            if ignore_masks:
                X_batch_masked[ignore_masks[idx] == 0] = 0
            X_batch_masked = X_batch_masked.double().to(device=config.device)
            H_batch = torch.sum(X_batch_masked.mT @ X_batch_masked, dim=0)
            H = H + H_batch
            del X_batch_masked, H_batch
        current_inps = layer_outputs
        layer.cpu()
        cleanup_memory()
    return H


@torch.no_grad()
def rotate_and_slice_global_pca(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
) -> None:
    """
    Rotate and slice using a single global PCA basis for all layers.
    """
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype
    
    layers = model_adapter.get_layers()
    slicing_scheduler.setup(
        hidden_size=model_adapter.hidden_size, 
        layers_num=len(layers), 
        parallel_blocks=model_adapter.parallel_blocks
    )
    
    # compute global covariance and eigendecomposition
    H = collect_global_covariance(model_adapter, dataloader, apply_mask)
    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).to(device=config.device)
    H[diag, diag] = H[diag, diag] + damp
    X_eig = torch.linalg.eigh(H)
    del H
    
    index = torch.argsort(X_eig[0], descending=True)
    Q_global = X_eig[1][:, index].to(device=config.device)
    
    if final_orientation == 'random':
        R = random_orthogonal_upper_left(Q_global.shape[0], slicing_scheduler.get_embedding_dimensions()[0])
        Q_global = Q_global @ R.to(Q_global.device, dtype=torch.float64)
    
    # rotate and slice embeddings
    rotate_embeddings(model_adapter, Q_global)
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    is_last_layer_special = not slicing_scheduler.do_slice_head
    
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Processing layers")):
        layer = layer_adapter.layer
        is_last_layer = (idx == len(layers) - 1)
        
        if not model_adapter.parallel_blocks:
            # Sequential blocks case (like OPT, LLaMA)
            new_attn_in_dim = slicing_scheduler.get_attention_input_dimension(idx)
            new_attn_out_dim = slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)
            new_mlp_in_dim = slicing_scheduler.get_mlp_input_dimension(idx)
            new_mlp_out_dim = slicing_scheduler.get_mlp_output_dimension(idx)

            # attention shortcut: None for global PCA
            layer.attn_shortcut_Q = None
            
            rotate_attention_inputs(layer_adapter, Q_global)
            slice_attention_inputs(layer_adapter, new_attn_in_dim)
            rotate_attention_output(layer_adapter, Q_global)
            slice_attention_output(layer_adapter, new_attn_out_dim)
            

            # mlp shortcut: None unless last layer with different output dimension
            if is_last_layer and is_last_layer_special and new_mlp_in_dim != new_mlp_out_dim:
                layer.mlp_shortcut_Q = nn.Parameter(
                    torch.eye(new_mlp_in_dim, new_mlp_out_dim, dtype=dtype, device='cpu')
                )
            else:
                layer.mlp_shortcut_Q = None

            rotate_mlp_input(layer_adapter, Q_global)
            slice_mlp_input(layer_adapter, new_mlp_in_dim)
            rotate_mlp_output(layer_adapter, Q_global)
            slice_mlp_output(layer_adapter, new_mlp_out_dim)
            
        else:
            new_in_dim = slicing_scheduler.get_attention_input_dimension(idx)
            new_out_dim = slicing_scheduler.get_mlp_output_dimension(idx)

            if is_last_layer and is_last_layer_special and new_in_dim != new_out_dim:
                layer.attn_shortcut_Q = nn.Parameter(
                    torch.eye(new_in_dim, new_out_dim, dtype=dtype, device='cpu')
                )
            else:
                layer.attn_shortcut_Q = None
            
            # Rotate all inputs and outputs with global Q
            rotate_attention_inputs(layer_adapter, Q_global)
            rotate_mlp_input(layer_adapter, Q_global)
            slice_attention_inputs(layer_adapter, new_in_dim)
            slice_mlp_input(layer_adapter, new_in_dim)
            
            rotate_attention_output(layer_adapter, Q_global)
            rotate_mlp_output(layer_adapter, Q_global)
            slice_attention_output(layer_adapter, new_out_dim)
            slice_mlp_output(layer_adapter, new_out_dim)
        
        layer.to('cpu')
        cleanup_memory()
    
    # Step 6: Rotate and slice head
    logging.info("Rotating head with global Q...")
    rotate_head(model_adapter, Q_global)
    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())
    
    # Store slicing configuration
    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()
    
    logging.info("=" * 60)
    logging.info("GLOBAL PCA: Rotation and slicing complete!")
    logging.info("MEMORY SAVINGS: Shortcut matrices eliminated (except last layer if needed)")
    logging.info("=" * 60)




# K-Block PCA: Groups of K layers share a single rotation matrix.


@torch.no_grad()
def rotate_and_slice_kblock(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    k_block: int,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
) -> dict:
    """
    Rotate and slice with K layers per block sharing a rotation matrix.
    """
    model = model_adapter.model
    model.eval()
    dtype = next(model.parameters()).dtype
    
    layers = list(model_adapter.get_layers())
    num_layers = len(layers)
    
    if k_block < 1 or k_block > num_layers:
        raise ValueError(f"k_block must be between 1 and {num_layers}, got {k_block}")
    
    # redirect to optimized implementations for edge cases
    if k_block == num_layers:
        rotate_and_slice_global_pca(
            model_adapter, dataloader, slicing_scheduler,
            apply_mask=apply_mask, final_orientation=final_orientation
        )
        return {"k_block": k_block, "num_blocks": 1, "shortcuts_stored": 1, "shortcuts_eliminated": num_layers * 2 - 1}

    if k_block == 1:
        rotate_and_slice_sequential(
            model_adapter, dataloader, slicing_scheduler,
            apply_mask=apply_mask, final_orientation=final_orientation
        )
        return {"k_block": 1, "num_blocks": num_layers, "shortcuts_stored": 2 * num_layers, "shortcuts_eliminated": 0}
    
    # Setup
    slicing_scheduler.setup(
        hidden_size=model_adapter.hidden_size,
        layers_num=num_layers,
        parallel_blocks=model_adapter.parallel_blocks
    )
    
    # calculate block structure
    num_full_blocks = num_layers // k_block
    num_blocks = num_full_blocks + (1 if num_layers % k_block > 0 else 0)
    
    block_ranges = []
    layer_to_block = {}
    for block_idx in range(num_blocks):
        start = block_idx * k_block
        end = min((block_idx + 1) * k_block, num_layers)
        block_ranges.append((start, end))
        for layer_idx in range(start, end):
            layer_to_block[layer_idx] = block_idx
    
    logging.info("=" * 60)
    logging.info(f"K-BLOCK PCA V2: K={k_block}")
    logging.info("=" * 60)
    logging.info(f"Layers: {num_layers}, Blocks: {num_blocks}")
    for i, (s, e) in enumerate(block_ranges):
        logging.info(f"  Block {i}: Layers {s}-{e-1}")
    
    # collect covariances for all blocks from original model
    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])
    
    block_covariances = [None] * num_blocks
    for idx, X_batch in enumerate(inps):
        X_batch_masked = X_batch.clone()
        if ignore_masks:
            X_batch_masked[ignore_masks[idx] == 0] = 0
        X_batch_masked = X_batch_masked.double().to(device=config.device)
        H_batch = torch.sum(X_batch_masked.mT @ X_batch_masked, dim=0)
        block_covariances[0] = H_batch if block_covariances[0] is None else block_covariances[0] + H_batch
        del X_batch_masked, H_batch
    
    current_inps = inps
    for layer_idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Collecting covariances")):
        layer = layer_adapter.layer.to(config.device)
        block_idx = layer_to_block[layer_idx]
        layer_outputs = []
        for i, inp in enumerate(current_inps):
            layer_args = layer_adapter.get_updated_args(inp, args[i])
            layer_args = map_tensors(layer_args, device=config.device)
            layer_kwargs = map_tensors(kwargs[i], device=config.device)
            
            out = layer(*layer_args, **layer_kwargs)
            if isinstance(out, tuple):
                out = out[layer_adapter.hidden_states_output_position]
            layer_outputs.append(out.cpu())
        for idx, X_batch in enumerate(layer_outputs):
            X_batch_masked = X_batch.clone()
            if ignore_masks:
                X_batch_masked[ignore_masks[idx] == 0] = 0
            X_batch_masked = X_batch_masked.double().to(device=config.device)
            H_batch = torch.sum(X_batch_masked.mT @ X_batch_masked, dim=0)
            if block_covariances[block_idx] is None:
                block_covariances[block_idx] = H_batch
            else:
                block_covariances[block_idx] = block_covariances[block_idx] + H_batch
            del X_batch_masked, H_batch
        
        current_inps = layer_outputs
        layer.cpu()
        cleanup_memory()
    
    # compute Q matrices for all blocks
    Q_blocks = []
    embedding_dim = slicing_scheduler.get_embedding_dimensions()[0]
    
    for block_idx in range(num_blocks):
        H_block = block_covariances[block_idx]

        damp = 0.01 * torch.mean(torch.diag(H_block))
        diag_idx = torch.arange(H_block.shape[-1]).to(device=config.device)
        H_block[diag_idx, diag_idx] = H_block[diag_idx, diag_idx] + damp
        
        X_eig = torch.linalg.eigh(H_block)
        index = torch.argsort(X_eig[0], descending=True)
        Q_block = X_eig[1][:, index].to(device=config.device, dtype=torch.float64)
        
        if final_orientation == 'random':
            R = random_orthogonal_upper_left(Q_block.shape[0], embedding_dim)
            Q_block = Q_block @ R.to(Q_block.device, dtype=torch.float64)
        
        Q_blocks.append(Q_block)
        block_covariances[block_idx] = None
    
    # apply rotations and slicing
    rotate_embeddings(model_adapter, Q_blocks[0])
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())
    
    shortcuts_stored = 0
    is_last_layer_special = not slicing_scheduler.do_slice_head
    
    for layer_idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):
        layer = layer_adapter.layer.to(config.device)
        block_idx = layer_to_block[layer_idx]
        Q_block = Q_blocks[block_idx]
        
        is_first_in_block = (layer_idx == block_ranges[block_idx][0])
        is_last_layer = (layer_idx == num_layers - 1)
        
        # Get dimensions
        new_attn_in_dim = slicing_scheduler.get_attention_input_dimension(layer_idx)
        new_attn_out_dim = slicing_scheduler.get_attention_output_dimension(layer_idx, match_head_dim=False)
        new_mlp_in_dim = slicing_scheduler.get_mlp_input_dimension(layer_idx)
        new_mlp_out_dim = slicing_scheduler.get_mlp_output_dimension(layer_idx)
        
        if not model_adapter.parallel_blocks:
            # block boundary shortcut
            if is_first_in_block and block_idx > 0:
                Q_prev = Q_blocks[block_idx - 1]
                shortcut = torch.matmul(Q_prev[:, :new_attn_in_dim].T, Q_block[:, :new_attn_in_dim])
                layer.attn_shortcut_Q = nn.Parameter(shortcut.to(dtype=dtype, device="cpu"))
                shortcuts_stored += 1
            else:
                layer.attn_shortcut_Q = None
            
            # Rotate and slice attention
            rotate_attention_inputs(layer_adapter, Q_block)
            slice_attention_inputs(layer_adapter, new_attn_in_dim)
            rotate_attention_output(layer_adapter, Q_block)
            slice_attention_output(layer_adapter, new_attn_out_dim)
            
            if is_last_layer and is_last_layer_special and new_mlp_in_dim != new_mlp_out_dim:
                layer.mlp_shortcut_Q = nn.Parameter(
                    torch.eye(new_mlp_in_dim, new_mlp_out_dim, dtype=dtype, device='cpu')
                )
                shortcuts_stored += 1
            else:
                layer.mlp_shortcut_Q = None
            
            # Rotate and slice MLP
            rotate_mlp_input(layer_adapter, Q_block)
            slice_mlp_input(layer_adapter, new_mlp_in_dim)
            rotate_mlp_output(layer_adapter, Q_block)
            slice_mlp_output(layer_adapter, new_mlp_out_dim)
            
        else:
            new_in_dim = slicing_scheduler.get_attention_input_dimension(layer_idx)
            new_out_dim = slicing_scheduler.get_mlp_output_dimension(layer_idx)

            if is_first_in_block and block_idx > 0:
                Q_prev = Q_blocks[block_idx - 1]
                shortcut = torch.matmul(Q_prev[:, :new_in_dim].T, Q_block[:, :new_in_dim])
                layer.attn_shortcut_Q = nn.Parameter(shortcut.to(dtype=dtype, device="cpu"))
                shortcuts_stored += 1
            elif is_last_layer and is_last_layer_special and new_in_dim != new_out_dim:
                layer.attn_shortcut_Q = nn.Parameter(
                    torch.eye(new_in_dim, new_out_dim, dtype=dtype, device='cpu')
                )
                shortcuts_stored += 1
            else:
                layer.attn_shortcut_Q = None
            
            rotate_attention_inputs(layer_adapter, Q_block)
            rotate_mlp_input(layer_adapter, Q_block)
            slice_attention_inputs(layer_adapter, new_in_dim)
            slice_mlp_input(layer_adapter, new_in_dim)
            rotate_attention_output(layer_adapter, Q_block)
            rotate_mlp_output(layer_adapter, Q_block)
            slice_attention_output(layer_adapter, new_out_dim)
            slice_mlp_output(layer_adapter, new_out_dim)
        
        layer.to("cpu")
        cleanup_memory()
    
    rotate_head(model_adapter, Q_blocks[-1])
    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())

    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()

    total_possible = 2 * num_layers if not model_adapter.parallel_blocks else num_layers
    eliminated = total_possible - shortcuts_stored
    logging.info(f"K-block rotation done. Shortcuts: {shortcuts_stored} stored, {eliminated} eliminated")
    
    return {
        "k_block": k_block,
        "num_blocks": num_blocks,
        "shortcuts_stored": shortcuts_stored,
        "shortcuts_eliminated": eliminated,
    }
