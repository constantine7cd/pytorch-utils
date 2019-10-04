import torch
import torch.nn.functional as F
import os


from metrics_check import check_error_equal_rate, check_error_equal_rate2


def checkpoint_(epoch, model, optimizer, path):
    """
        Read checkpoint example:

        state = torch.load(filepath)

        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
    """

    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(state, path)


def ramp_up(epoch, range_=80, gauss_factor=-5):
    if epoch > range_:
        return torch.tensor(1).type(torch.float32)

    t = epoch / range_

    return torch.exp(gauss_factor * torch.mul(1. - t, 1. - t)).type(torch.float32)


def train_(model, optimizer, train_dset, val_dset, train_eer_dset, device, epochs=5, print_every=5):
    train_history = {
        "loss": [],
        "eer_train": [],
        "eer_val": []
    }

    model = model.to(device=device)

    for e in range(epochs):
        for t, (x, y) in enumerate(train_dset):
            model.train()
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)

            scores = model(x)
            loss = F.binary_cross_entropy_with_logits(scores, y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_error_equal_rate(val_dset, model, verbose=True)
                print()

            train_history['loss'].append(loss.item())

        train_history['eer_train'].append(check_error_equal_rate(train_eer_dset, model, device))
        train_history['eer_val'].append(check_error_equal_rate(val_dset, model, device))

    return train_history


def loss_val(model, val_dset, device):
    model.eval()

    with torch.no_grad():
        x, y = next(iter(val_dset))

        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)

        scores = model(x)

        loss = F.binary_cross_entropy_with_logits(scores, y)

    return loss.item()


def train_pi_model(model, optimizer, train_dset, val_dset, train_dset_eer, device, epochs=5, print_every=5):
    train_history = {
        "loss_train": [],
        "loss_val": [],
        "eer_train": [],
        "eer_val": []
    }

    model = model.to(device=device)

    for e in range(epochs):

        ramp_up_value = ramp_up(e)
        ramp_up_value = ramp_up_value.to(device=device, dtype=torch.float32)

        for t, (x1, x2, y, mask) in enumerate(train_dset):
            model.train()

            x1 = x1.to(device=device, dtype=torch.float32)
            x2 = x2.to(device=device, dtype=torch.float32)

            y = y.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.uint8)

            scores1 = model(x1)
            scores2 = model(x2)

            cross_entropy = F.binary_cross_entropy_with_logits(scores1[mask == 1], y[mask == 1])

            mse = ramp_up_value * F.mse_loss(scores1, scores2)

            loss = cross_entropy + mse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % print_every == 0:
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                check_error_equal_rate(val_dset, model, device=device, verbose=True)
                print()

                train_history['loss_train'].append(loss.item())
                train_history['loss_val'].append(loss_val(model, val_dset, device))

        train_history['eer_train'].append(check_error_equal_rate(train_dset_eer, model, device))
        train_history['eer_val'].append(check_error_equal_rate(val_dset, model, device))

    return train_history


def loss_val_v2(model, val_dset, device):
    model.eval()

    with torch.no_grad():
        x, y = next(iter(val_dset))

        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)

        _, scores = model(x)

        loss = F.binary_cross_entropy_with_logits(scores, y)

    return loss.item()


def train_pi_model_v2(model, optimizer, train_dset, val_dset, train_dset_eer, device, start_e=0, epochs=5,
                      print_every=5):
    train_history = {
        "loss_train": [],
        "loss_val": [],
        "eer_train": [],
        "eer_val": []
    }

    model = model.to(device=device)

    for e in range(start_e, epochs):

        ramp_up_value = ramp_up(e)
        ramp_up_value = ramp_up_value.to(device=device, dtype=torch.float32)

        for t, (x1, x2, y, mask) in enumerate(train_dset):
            model.train()

            x1 = x1.to(device=device, dtype=torch.float32)
            x2 = x2.to(device=device, dtype=torch.float32)

            y = y.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.uint8)

            encodings1, scores1 = model(x1)
            encodings2, scores2 = model(x2)

            cross_entropy = F.binary_cross_entropy_with_logits(scores1[mask == 1], y[mask == 1])

            mse = ramp_up_value * F.mse_loss(encodings1, encodings2)

            loss = cross_entropy + mse

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if t % print_every == 0:
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                check_error_equal_rate2(val_dset, model, device=device, verbose=True)
                print()

                train_history['loss_train'].append(loss.item())
                train_history['loss_val'].append(loss_val_v2(model, val_dset, device))

        train_history['eer_train'].append(check_error_equal_rate2(train_dset_eer, model, device))
        train_history['eer_val'].append(check_error_equal_rate2(val_dset, model, device))

    return train_history


def train_pi_model_v3(model, optimizer, train_dset, val_dset, train_dset_eer, device, checkpoint_params,
                      start_e=0, epochs=5, checkpoint=True, checkpoint_every=10):
    lr, dr, checkpoint_path = checkpoint_params

    train_history = {
        "loss_train": [],
        "loss_val": [],
        "eer_train": [],
        "eer_val": []
    }

    model = model.to(device=device)

    for e in range(start_e, epochs):

        ramp_up_value = ramp_up(e)
        ramp_up_value = ramp_up_value.to(device=device, dtype=torch.float32)

        for t, (x1, x2, y, mask) in enumerate(train_dset):
            model.train()

            x1 = x1.to(device=device, dtype=torch.float32)
            x2 = x2.to(device=device, dtype=torch.float32)

            y = y.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.uint8)

            encodings1, scores1 = model(x1)
            encodings2, scores2 = model(x2)

            cross_entropy = F.binary_cross_entropy_with_logits(scores1[mask == 1], y[mask == 1])

            mse = ramp_up_value * F.mse_loss(encodings1, encodings2)

            loss = cross_entropy + mse

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        if checkpoint and e % checkpoint_every == 0:
            checkpoint_(e, model, optimizer,
                        path=os.path.join(checkpoint_path, str(lr) + "_" + str(dr)[2:6] + "_" + str(e) + '.pt'))

            print(f"Checkpoint was created at epoch: {e}", end="\n")

        train_history['loss_train'].append(loss.item())
        train_history['loss_val'].append(loss_val_v2(model, val_dset, device))

        train_history['eer_train'].append(check_error_equal_rate2(train_dset_eer, model, device))
        train_history['eer_val'].append(check_error_equal_rate2(val_dset, model, device))

    checkpoint_(e, model, optimizer,
                path=os.path.join(checkpoint_path, str(lr) + "_" + str(dr)[2:6] + str(e) + '.pt'))
    print()

    return train_history
