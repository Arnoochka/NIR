import os
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, train
from mindspore.communication import init

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, pipeline_stages=1)
init()
ms.set_seed(1)

class Network(nn.Cell):
    """Network"""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Dense(28*28, 512)

    def construct(self, x):
        x = self.flatten(x)
        return x

net = Network()
net.layer1.pipeline_stage = 0

def create_dataset(batch_size):
    """create dataset"""
    dataset_path = os.getenv("DATA_PATH")
    dataset = ds.MnistDataset(dataset_path)
    image_transforms = [
        ds.vision.Rescale(1.0 / 255.0, 0),
        ds.vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        ds.vision.HWC2CHW()
    ]
    label_transform = ds.transforms.TypeCast(ms.int32)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

data_set = create_dataset(32)

optimizer = nn.SGD(net.trainable_params(), 1e-2)
loss_fn = nn.CrossEntropyLoss()

loss_cb = train.LossMonitor()
net_with_grads = nn.PipelineCell(nn.WithLossCell(net, loss_fn), 16)
model = ms.Model(net_with_grads, optimizer=optimizer)
model.train(1, data_set, callbacks=[loss_cb], dataset_sink_mode=True)

"""
В этом коде используется разделение на 4 MicroBatch. При этом используется конвеер всего из 2-х карт. 
"""


