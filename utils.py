import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

def calculate_accuracy_for_time(pred, real, alpha=60*60):
    ''' Calculate accuracy for two times
        If the difference between the two times is less than the reference time (alpha),
        the two times are considered to be consistent.
    '''
    if pred.shape != real.shape:
        print(pred.shape, real.shape)
        raise Exception("Two tensors have to be has same shape")
    alpha_per_day = alpha / float(24*60*60)
    pred = pred.flatten();      real = real.flatten()
    correct = (torch.abs(pred - real) <= alpha_per_day).sum().item()
    return correct / float(len(pred))

def calculate_accuracy_for_event(pred, real, device, alpha_heater = 1):

    if pred.shape != real.shape:
        print(pred.shape, real.shape)
        raise Exception("Two tensors have to be has same shape")

    alpha = alpha_heater / 50.

    pred = pred.flatten();  real = real.flatten()

    correct_heater = torch.abs(pred-real) <= alpha
    correct_light = torch.eq(pred.round().int(), real.round().int())

    correct_total = torch.where(device <= 3, correct_heater, correct_light)              # 0~3: heaters  // 4~7: lights
    accuracy = correct_total.sum().item() / float(len(correct_total))

    return accuracy


def train_one_epoch(model, dataloader, optimizer, losses_weight=[0, 1, 0]):
    
    train_losses = {
        'total_loss': 0.,
        'time_loss': 0.,
        'device_loss': 0.,
        'event_loss': 0.
    }
    
    model.train()
    criterion_time = nn.MSELoss()
    criterion_device = nn.CrossEntropyLoss()
    criterion_event = nn.MSELoss()

    for inp, (time, device, event) in dataloader:
        inp = inp.to(model.device)
        time = time.to(model.device);   device = device.to(model.device);   event = event.to(model.device)

        time_predicted, device_predicted, event_predicted = model(inp)
        loss_time = criterion_time(time_predicted, time)
        loss_device = criterion_device(device_predicted, device)
        loss_event = criterion_event(event_predicted, event)

        loss = losses_weight[0] * loss_time + losses_weight[1] * loss_device + losses_weight[2] * loss_event
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses['total_loss'] += loss.item() * inp.size(0)
        train_losses['time_loss'] += loss_time.item() * inp.size(0)
        train_losses['device_loss'] += loss_device.item() * inp.size(0)
        train_losses['event_loss'] += loss_event.item() * inp.size(0)

    train_losses['total_loss'] /= len(dataloader.dataset)
    train_losses['time_loss'] /= len(dataloader.dataset)
    train_losses['device_loss'] /= len(dataloader.dataset)
    train_losses['event_loss'] /= len(dataloader.dataset)

    return train_losses


def test(model, dataloader):

    times_predicted, devices_predicted, events_predicted = torch.tensor([]), torch.tensor([]), torch.tensor([])
    times_real, devices_real, events_real = torch.tensor([]), torch.tensor([]), torch.tensor([])

    model.eval()
    with torch.no_grad():
        for i, (inp, time, device, event, device_mask) in enumerate(dataloader):
            inp = inp.to(model.device)
            device_mask = device_mask.to(model.device)

            time_predicted, device_predicted, event_predicted = model(inp)
            device_predicted = device_predicted * device_mask


            device_predicted = torch.argmax(device_predicted, dim=1)

            times_predicted = torch.cat((times_predicted, time_predicted.cpu()), dim=0)
            devices_predicted = torch.cat((devices_predicted, device_predicted.cpu()), dim=0)
            events_predicted = torch.cat((events_predicted, event_predicted.cpu()), dim=0)

            times_real = torch.cat((times_real, time.cpu()), dim=0)
            devices_real = torch.cat((devices_real, device.cpu()), dim=0)
            events_real = torch.cat((events_real, event.cpu()), dim=0)


    accuracy_time = calculate_accuracy_for_time(times_predicted, times_real)
    accuracy_device = accuracy_score(devices_predicted.detach().numpy(), devices_real.detach().numpy())
    accuracy_event = calculate_accuracy_for_event(events_predicted, events_real, devices_real, alpha_heater=1)

    f1_device = f1_score(devices_predicted.detach().numpy(), devices_real.detach().numpy(), average='micro')
    
    return {
        'accuracy_time': accuracy_time,
        'accuracy_device': accuracy_device,
        'accuracy_event': accuracy_event,
        'f1_device': f1_device
    }
