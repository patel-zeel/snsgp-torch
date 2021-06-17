import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
from varz.tensorflow import Vars, minimise_l_bfgs_b, minimise_adam
from stheno import GP, EQ
import lab.tensorflow as B

# Sample a true, underlying function and observations with known noise.
x_obs = B.rand(500, 2).reshape(-1, 2)
f_fixed = GP(3 * EQ().stretch([0.2, 0.8]))
noise_true = 0.01
f_true, y_obs = f_fixed.measure.sample(f_fixed(x_obs), f_fixed(x_obs, noise_true))

# Construct a model with learnable parameters.
def model(vs):
    kernel = vs.positive(1., name="std") * EQ().stretch(vs.positive(shape=(1,2), name="ls"))
    noise = vs.positive(0.1, name="noise")
    return GP(kernel), noise

# Define an objective function.
def objective(vs):
    f, noise = model(vs)
    return -f(x_obs, noise).logpdf(y_obs)

# Perform optimisation and print the learned parameters.
vs = Vars(tf.float32)
minimise_l_bfgs_b(objective, vs, trace=True, jit=False, iters=10000)
vs.print()