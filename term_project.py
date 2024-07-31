import torch
import torch.distributed as dist
import os
import sys
import socket
import traceback
import datetime
import torch.nn as nn
import torch.optim as optim
from random import Random
import pandas as pd
import numpy as np
import scipy.stats
import csv

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(12, 30)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(30, 40)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(40, 1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def percent_difference(target, prediction):
    if target == 0:
        if prediction == 0:
            return torch.tensor(0.0)
        else:
            return torch.tensor(float('inf'))
    else:
        return ((prediction - target) / target) * 100

"""Distributed Synchronous SGD Example"""
def run(rank, world_size):
    # torch.manual_seed(1234)

    data_frame_from_csv = pd.read_csv('water_quality_and_weather.csv')


    data_frame_from_csv_tuples = list(data_frame_from_csv.itertuples(index=False, name=None))

    #0 station location
    #1 date
    #2 time
    #3 entero
    #4 water_temp
    #5 do
    #6 ph
    #7 chlorophyll
    #8 density
    #9 fecal
    #10 salinity
    #11 air_temp
    #12 humidity
    #13 windspeed 
    #14 cloud_cover
    #15 solar_radiation

    columns_as_floats = [(float(line[3]), float(line[4]), float(line[5]),
                        float(line[6]), float(line[7]), float(line[8]), 
                        float(line[9]), float(line[10]), float(line[11]),
                        float(line[12]), float(line[13]), float(line[14]), 
                        float(line[15])) for line in data_frame_from_csv_tuples]
    

    #0 entero
    #1 water_temp
    #2 do
    #3 ph
    #4 chlorophyll
    #5 density
    #6 fecal
    #7 salinity
    #8 air_temp
    #9 humidity
    #10 windspeed 
    #11 cloud_cover
    #12 solar_radiation

    feature_target_tensor_tuple = [(torch.tensor((line[0], line[1], line[2], 
                                                  line[3], line[4], line[5], 
                                                  line[6], line[8], line[9], 
                                                  line[10], line[11], line[12])),
            torch.tensor(line[7]).reshape(1)) for line in columns_as_floats]

    size = dist.get_world_size() + 1

    partition_sizes = [1.0 / size for _ in range(size)]
    partition_from_partitioner = DataPartitioner(feature_target_tensor_tuple, partition_sizes)

    partition = partition_from_partitioner.use(dist.get_rank())
    test_partition = partition_from_partitioner.use(size - 1)

    train_set = torch.utils.data.DataLoader(partition, shuffle=True)
    test_set = torch.utils.data.DataLoader(test_partition, shuffle=True)


    model = nn.parallel.DistributedDataParallel(Net()).float()
        # model = load_model(nn.parallel.DistributedDataParallel(Net()), "best_model.pth").float()

    optimizer = optim.SGD(model.parameters(), lr=0.000001, momentum=0.5)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    print("Proceeding with training...")
    epoch_losses = []
    epochs = []    
    for epoch in range(1000):
        epoch_loss = 0.0
        for i, (data, target) in enumerate(train_set):
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()


        if dist.get_rank() == 0:
            print(epoch_loss)
            epochs.append(epoch+1)
            epoch_losses.append(epoch_loss)
        if dist.get_rank() == 0 and epoch_loss < best_loss:
            best_loss = epoch_loss

    print("Done training, now predictions")
    if (dist.get_rank() == 0):
        percent_differences = []
        actuals = []
        predictions = []

        for i, (data, target) in enumerate(test_set):

            #the [0][0] just get's the value embedded in the returned tensor, it doesn't do any data manipulation
            prediction = model(data)[0][0]
            predictions.append(prediction)
            actual = target[0][0]
            actuals.append(actual)
            percent_differences.append(np.abs(percent_difference(actual, prediction).item()))

        average_percent_difference = sum(percent_differences) / len(percent_differences)

        print("Average percent difference between predicted and actual water salinity:", average_percent_difference)
        print("Best epoch's MSE: ", best_loss)

        #write out data to csv to create a plot
        # with open('epochs_vs_loss.csv', 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(['Epoch', 'MSE Loss'])
        #     writer.writerows(zip(epochs, epoch_losses))

    print("Calculating correlation statistics")
    data_inputs_for_pearson_calculation = []
    predicted_outputs_for_pearson_calculation = []

    for j, (data, target) in enumerate(train_set):

        data_inputs_for_pearson_calculation.append(data)
        predicted_output_for_pearson_calculation = model(data)
        predicted_outputs_for_pearson_calculation.append(predicted_output_for_pearson_calculation.squeeze())
        
    #processing to get x and y to same shape and data structure type for Pearson correlation coefficient
    data_inputs_as_tensor = torch.stack(data_inputs_for_pearson_calculation)
    data_outputs_as_tensor = torch.stack(predicted_outputs_for_pearson_calculation)

    data_inputs_as_array = data_inputs_as_tensor.numpy()
    data_inputs_as_array = data_inputs_as_array.squeeze(axis=1)

    data_outputs_as_array = data_outputs_as_tensor.detach().numpy()
    data_outputs_as_array = data_outputs_as_array.flatten()

    correlations = []
    for i in range(data_inputs_as_array.shape[1]):
        feature_data = data_inputs_as_array[:, i]
        corr, _ = scipy.stats.pearsonr(feature_data, data_outputs_as_array)
        correlations.append(corr)
        print(f"Feature {i} correlation with output: {corr:.3f}")



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'columbia'
    os.environ['MASTER_PORT'] = '30905'
            

    #initialize the process group
    dist.init_process_group("gloo", rank=int(rank), world_size=int(world_size), init_method='tcp://columbia:23456', timeout=datetime.timedelta(weeks=120))


if __name__ == "__main__":
    print("Starting program...")
    try:
        setup(sys.argv[1], sys.argv[2])
        print(socket.gethostname() + ": Setup completed!")
        run(int(sys.argv[1]), int(sys.argv[2]))
    except Exception as e:
        traceback.print_exc()
        sys.exit(3)