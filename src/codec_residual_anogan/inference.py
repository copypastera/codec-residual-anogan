"""AnoGAN latent inversion and anomaly scoring."""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


def _components(real, generated, discriminator, real_feature=None):
    reconstruction_l1 = torch.mean(torch.abs(real - generated), dim=1)
    reconstruction_mse = torch.mean(torch.square(real - generated), dim=1)
    reconstruction_smooth_l1 = functional.smooth_l1_loss(
        generated, real, reduction="none").mean(dim=1)
    if real_feature is None:
        real_feature = discriminator.feature(real)
    generated_feature = discriminator.feature(generated)
    feature_score = torch.mean(
        torch.abs(real_feature - generated_feature), dim=1)
    return (
        reconstruction_l1, reconstruction_mse,
        reconstruction_smooth_l1, feature_score)


def _objective(components, score_config):
    reconstruction = score_config.get("reconstruction_loss", "l1")
    index = {"l1": 0, "mse": 1, "smooth_l1": 2}[reconstruction]
    reconstruction_weight = float(
        score_config.get("reconstruction_weight", 0.5))
    feature_weight = float(
        score_config.get(
            "discriminator_feature_weight", 1.0 - reconstruction_weight))
    return (
        reconstruction_weight * components[index]
        + feature_weight * components[3])


def invert_features(generator, discriminator, features, config, device,
                    seed, status_callback=None, random_state=None):
    """Optimize one latent vector per feature row and return components."""
    generator.eval()
    discriminator.eval()
    inference = config["inference"]
    score_config = config["anomaly_score"]
    latent_dim = int(config["model"]["latent_dim"])
    steps = int(inference["latent_optimization_steps"])
    restarts = int(inference.get("latent_restarts", 1))
    learning_rate = float(inference["latent_learning_rate"])
    batch_size = int(inference.get("batch_size", 512))
    optimizer_name = inference.get("latent_optimizer", "adam").lower()
    if steps <= 0 or restarts <= 0 or batch_size <= 0:
        raise ValueError("steps, restarts, and batch size must be positive")
    if optimizer_name not in ("adam", "adamw"):
        raise ValueError("latent optimizer must be adam or adamw")

    names = (
        "reconstruction_l1", "reconstruction_mse",
        "reconstruction_smooth_l1", "discriminator_feature_score",
        "initial_latent_loss", "final_latent_loss",
        "loss_reduction", "selected_restart")
    output = {
        name: np.empty(
            len(features),
            dtype=np.int16 if name == "selected_restart" else np.float32)
        for name in names}
    if random_state is None:
        random_state = np.random.RandomState(seed)
    total_batches = int(math.ceil(len(features) / float(batch_size)))
    old_generator_flags = [
        parameter.requires_grad for parameter in generator.parameters()]
    old_discriminator_flags = [
        parameter.requires_grad for parameter in discriminator.parameters()]
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)

    started = time.time()
    try:
        for batch_index, start in enumerate(
                range(0, len(features), batch_size), 1):
            stop = min(start + batch_size, len(features))
            batch = stop - start
            base = torch.from_numpy(
                np.ascontiguousarray(features[start:stop])).to(device)
            real = base[:, None, :].expand(
                batch, restarts, base.shape[1]).reshape(
                    batch * restarts, base.shape[1])
            with torch.no_grad():
                base_feature = discriminator.feature(base)
                real_feature = base_feature[:, None, :].expand(
                    batch, restarts, base_feature.shape[1]).reshape(
                        batch * restarts, base_feature.shape[1])
            initial = random_state.standard_normal(
                (batch * restarts, latent_dim)).astype(np.float32)
            latent = nn.Parameter(torch.from_numpy(initial).to(device))
            optimizer_class = (
                torch.optim.AdamW if optimizer_name == "adamw"
                else torch.optim.Adam)
            optimizer = optimizer_class([latent], lr=learning_rate)
            initial_loss = None
            for step in range(steps + 1):
                generated = generator(torch.tanh(latent))
                components = _components(
                    real, generated, discriminator, real_feature)
                losses = _objective(components, score_config)
                if step == 0:
                    initial_loss = losses.detach().reshape(batch, restarts)
                if step == steps:
                    break
                optimizer.zero_grad(set_to_none=True)
                losses.mean().backward()
                optimizer.step()

            final_loss = losses.detach().reshape(batch, restarts)
            selected = torch.argmin(final_loss, dim=1)
            rows = torch.arange(batch, device=device)
            selected_components = [
                value.detach().reshape(batch, restarts)[rows, selected]
                .cpu().numpy()
                for value in components]
            selected_initial = initial_loss[rows, selected].cpu().numpy()
            selected_final = final_loss[rows, selected].cpu().numpy()
            output["reconstruction_l1"][start:stop] = selected_components[0]
            output["reconstruction_mse"][start:stop] = selected_components[1]
            output["reconstruction_smooth_l1"][start:stop] = (
                selected_components[2])
            output["discriminator_feature_score"][start:stop] = (
                selected_components[3])
            output["initial_latent_loss"][start:stop] = selected_initial
            output["final_latent_loss"][start:stop] = selected_final
            output["loss_reduction"][start:stop] = (
                selected_initial - selected_final)
            output["selected_restart"][start:stop] = (
                selected.cpu().numpy().astype(np.int16))
            if status_callback:
                status_callback(
                    batch_index, total_batches, stop,
                    time.time() - started)
    finally:
        for parameter, flag in zip(
                generator.parameters(), old_generator_flags):
            parameter.requires_grad_(flag)
        for parameter, flag in zip(
                discriminator.parameters(), old_discriminator_flags):
            parameter.requires_grad_(flag)
    return output, time.time() - started


def anomaly_scores(components, config):
    score_config = config["anomaly_score"]
    if score_config.get("component_normalization", "none") != "none":
        raise ValueError("the released best pipeline uses no score normalization")
    reconstruction_name = {
        "l1": "reconstruction_l1",
        "mse": "reconstruction_mse",
        "smooth_l1": "reconstruction_smooth_l1",
    }[score_config.get("reconstruction_loss", "l1")]
    reconstruction_weight = float(
        score_config.get("reconstruction_weight", 0.5))
    feature_weight = float(
        score_config.get(
            "discriminator_feature_weight", 1.0 - reconstruction_weight))
    return np.asarray(
        reconstruction_weight * components[reconstruction_name]
        + feature_weight * components["discriminator_feature_score"],
        dtype=np.float32)
