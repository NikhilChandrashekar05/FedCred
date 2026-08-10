"PrivacyEngine wraps the model, optimizer and dataloader & will auto handle gradient clipping, and noise injection. Through this the privacy is maintained"

"Opacus needs to compute per sample gradients, which means it processes each training example individually before aggregating. "
"This uses more memory than standard batch training. BatchMemoryManager handles that memory efficiently so i don't run out of RAM on large datasets."

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import torch

"The setup function needed before training. "
"""
- make_private_with_epsilon is important because tells target epsilon &  cals how much noise to add each step for privacy, Opacus does this
- target_delta is a secondary privacy paremeter(1/len of dataset). Shows prob. that privacy fails
- max grad norm is the clipping threshold, gradients get clipped to this norm before noise is added
- epochs is meant for Opcaus figuring total training epochs to calc noise level needed to hit target epsilon
"""

#4 privacy levels for training, generates privacy utility tradeoff maybe for UI
EPSILON_LEVELS = [1, 5, 10, float('inf')]

def make_private(model, optimizer, data_loader, target_epsilon, target_delta, max_grad_norm, epochs):
    # create the Opacus privacy engine
    privacy_engine = PrivacyEngine()

    # attach the privacy engine to the model, optimizer, and dataloader
    # this transforms standard training into DP training
    # Opacus replaces the optimizer and dataloader with privacy-aware versions
    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
        epochs=epochs,
    )

    return model, optimizer, data_loader, privacy_engine

#Full private training loop, Opacus wraps everything so code doesnt need to change, client would use this funcitonality when DP is turned on
def train_with_privacy(model, optimizer, data_loader, criterion, epsilon, delta=None, max_grad_norm=1.0, epochs=3):

    # set default delta to 1/dataset_size if not provided 
    if delta is None:
        delta = 1 / len(data_loader.dataset)

    model.train()

    # wrap model, optimizer, and dataloader with Opacus privacy engine
    model, optimizer, data_loader, privacy_engine = make_private(
        model=model,
        optimizer=optimizer,
        data_loader=data_loader,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=max_grad_norm,
        epochs=epochs,
    )

    # training loop identical to standard training
    # Opacus handles the privacy math automatically 
    model.train()

    for epoch in range(epochs):
        # BatchMemoryManager handles the extra memory Opacus needs
        # for per-sample gradient computation on large datasets
        with BatchMemoryManager(
            data_loader=data_loader,
            max_physical_batch_size=64,
            optimizer=optimizer
        ) as memory_safe_loader:

            for X_batch, y_batch in memory_safe_loader:
                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()

    # report the actual epsilon achieved after training
    actual_epsilon = privacy_engine.get_epsilon(delta)
    print(f"Differential Privacy training complete — ε = {actual_epsilon:.2f}, δ = {delta:.2e}")

    return model, actual_epsilon


def get_privacy_settings(epsilon):
    # if epsilon is infinity, no privacy — return None to signal standard training
    if epsilon == float('inf'):
        return None

    # otherwise return the DP config for this epsilon level
    return {
        'target_epsilon': epsilon,
        'max_grad_norm': 1.0,
        'epochs': 3,
    }