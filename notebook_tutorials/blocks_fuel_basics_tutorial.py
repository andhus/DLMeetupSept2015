
# coding: utf-8

# # Linnear Regresion in Blocks

# In[1]:

from __future__ import division

import numpy as np


# ## Dataset

# In[2]:

from fuel.datasets import MNIST
mnist = MNIST("train")


# In[3]:

mnist.num_examples


# In[4]:

mnist.sources


# In[5]:

handle = mnist.open()
data_sample = mnist.get_data(handle, [0, 1, 2])  # (ndarray, dnarray)


# In[6]:

data_sample[0].shape  # features


# In[7]:

data_sample[1].shape  # targets


# ## DataStream

# In[8]:

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

data_stream = DataStream.default_stream(
    mnist,
    iteration_scheme=SequentialScheme(
        mnist.num_examples,
        batch_size=256
    )
)


# In[9]:

data_stream.sources


# In[10]:

epoch = data_stream.get_epoch_iterator()
batch = next(epoch)  # (ndarray, dnarray)


# In[11]:

batch[0].shape


# In[12]:

batch[1].shape


# ## Transformers

# In[13]:

from fuel.transformers import Flatten


# In[14]:

data_stream = Flatten(data_stream)


# In[15]:

epoch = data_stream.get_epoch_iterator()
batch = next(epoch)  # (ndarray, dnarray)


# In[16]:

batch[0].shape


# In[17]:

batch[1].shape


# ## Model / Bricks

# In[18]:

from blocks.bricks import Linear
from blocks import initialization as init


# ## \# 1 Configuration

# In[19]:

linear = Linear(
    input_dim=28*28,
    output_dim=10,
    weights_init=init.IsotropicGaussian(0.01),
    biases_init=init.Constant(0)
)


# ## \# 2 Allocation (Optional)

# In[20]:

linear.params


# In[21]:

linear.allocate()


# In[22]:

linear.params


# ## \# 3 Initialization

# In[23]:

W = linear.params[0]
W.eval()


# In[24]:

linear.initialize()


# In[25]:

W = linear.params[0]
W.eval()


# ##  \# 4 Application

# In[26]:

from theano import tensor

X = tensor.fmatrix('X')


# In[27]:

y_hat = linear.apply(X)


# In[28]:

type(y_hat)


# In[29]:

y_hat


# ## Building your own Bricks

# In[30]:

from blocks.bricks import Initializable
from blocks.bricks import Softmax

class SoftmaxLinear(Initializable):
    
    def __init__(
        self,
        input_dim,
        output_dim,
        **kwargs
    ):
        super(SoftmaxLinear, self).__init__(**kwargs)
        self.linear = Linear(
            input_dim=input_dim,
            output_dim=output_dim
        )
        self.sofmax = Softmax()
        
        self.children = [
            self.linear,
            self.sofmax
        ]
        
    def apply(self, input_):
        output = self.sofmax.apply(
            self.linear.apply(
                input_
            )
        )
        return output


# In[31]:

softmax_linear = SoftmaxLinear(
    28*28,
    10,
    weights_init=init.IsotropicGaussian(0.01),
    biases_init=init.Constant(0)
)


# In[32]:

softmax_linear.initialize()
softmax_linear.linear.params


# ## Cost

# In[33]:

X = tensor.matrix('features')  # match sources in datastream
y = tensor.lmatrix('targets')


# In[34]:

data_stream.sources


# In[35]:

y_hat = softmax_linear.apply(X)


# In[1]:

# Run this to apply an MLP instead of Linear Regression

# from blocks.bricks import MLP, Tanh, Softmax

# mlp = MLP(
#     activations=[Tanh(), None],
#     dims=[28*28, 200, 10],
#     weights_init=init.IsotropicGaussian(0.01),
#     biases_init=init.Constant(0)
# )
# mlp.initialize()

# y_hat = Softmax().apply(mlp.apply(X))


# In[ ]:

from blocks.bricks.cost import CategoricalCrossEntropy

cost = CategoricalCrossEntropy().apply(y=y.flatten(), y_hat=y_hat)
cost.name = 'cost'


# ## Computational (Annotated) Graph

# In[ ]:

from blocks.graph import ComputationGraph

cg = ComputationGraph(cost)
cg.parameters


# In[ ]:

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT

weights = VariableFilter(roles=[WEIGHT])(cg.variables)
weights


# ## Training Algorithm

# In[ ]:

from blocks.algorithms import GradientDescent, Scale

algorithm = GradientDescent(
    cost=cost,
    params=cg.parameters,
    step_rule=Scale(learning_rate=0.1)
)


# In[ ]:

from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.plot import Plot
from blocks.extensions.monitoring import TrainingDataMonitoring


main_loop = MainLoop(
    data_stream=data_stream,
    algorithm=algorithm,
    extensions=[
        FinishAfter(after_n_epochs=100),
        TrainingDataMonitoring([cost], after_epoch=True),
        Plot(
            document='blocks_fuel_basics_tutorial LINEAR REG',
            channels=[['cost']],
            after_epoch=True
        ),
        Printing()
    ]
)


# In[ ]:

main_loop.run()


# In[ ]:



