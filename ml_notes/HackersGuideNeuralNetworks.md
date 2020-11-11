# Hacker's Guide to Neural Networks

## Chapter 1: Real-Valued Circuits

### Base Case: Single Gate in the Circuit

<img src="figures/image-20201109064934397.png" alt="image-20201109064934397" style="zoom:50%;" />

```python
def forwardMultiplyGate(x,y):
  return x*y

assert(forwardMultiplyGate(-2,3) == -6)
```

$$
f(x,y) = xy
$$

* In this example: the gate takes two inputs and produces a **single** output

#### The Goal

The problem looks like this:

1. We provide the circuit specific input values
2. The circuit computes the output value
3. *How do you change the input slightly to increase the output?* 

#### Strategy #1: Random Local Search

* We can easily "forward" (i.e. compute the output) of the circuit for any `x` and `y`, so we can randomly tweak `x` and `y` and keep track of the best tweak.

  ```python
  import numpy as np
  
  best_out = float('-inf')
  tweak_amount = 0.01
  num_iter = 100
  
  # inputs
  x, y = -2, 3
  # keeping track of best inputs
  best_x, best_y = x, y
  
  for i in range(num_iter):
      x_try = x + tweak_amount * (np.random.random() * 2 - 1) # tweak x a bit
      y_try = y + tweak_amount * (np.random.random() * 2 - 1) # tweak y a bit
      out = forwardMultiplyGate(x_try, y_try)
      if out > best_out:
          best_out = out
          best_x, best_y = x_try, y_try
  ```

* This strategy works, but it is inefficient

  * How would you do this with millions of inputs?

#### Strategy #2: Numerical Gradient

* *Intuition*: Imagine taking the output value from the circuit and tugging on it in the positive direction. This induces forces on the inputs `x` and `y`. These forces tell us how to change `x` and `y` to increase the output value

> The derivative can be thought of as a force on each input as we pull on the output to become higher.

* Idea: rather than looking backwards through the circuit, change the input slightly and see how the output changes. This is the derivative
  $$
  \frac{\partial f(x,y)}{\partial x} = \frac{f(x+h,y) - f(x,y)}{h}
  $$
  where $h$ is small (the tweak amount).

  * Numerator on the RHS measures the difference in the output due to the tweak of the input
  * The denominator on the RHS normalizes the difference by the arbitrary tweak amount

  * On the LHS of the above, it is not division. The entire entity is one thing: the partial derivative (i.e. the derivative of f(x,y) w.r.t. x). The RHS is division.

  ```python
  x, y = -2, 3
  out = forwardMultiplyGate(x, y)
  h = 0.0001
  
  # compute the derivative wrt x
  xph = x + h
  out2 = forwardMultiplyGate(xph,y)
  x_derivative = (out2 - out) / h
  # compute the derivative wrt y
  yph = y + h
  out3 = forwardMultiplyGate(x,yph)
  y_derivative = (out3 - out) / h
  ```

  * Ideally h would be infinitesimally small

> The derivative with respect to some input can be computed by tweaking that input by a small amount and observing the change on the output value.

* The **gradient** is made up of the derivatives of all the inputs concatenated in a vector.

* If we let the inputs respond to the tug by following the gradient a tinay amount (i.e. we just add the derivative on top of every input), we increase the output value:

  ```python
  step_size = 0.01
  out = forwardMultiplyGate(x,y)
  x = x + step_size * x_derivative
  y = y + step_size * y_derivative
  out_new = forwardMultiplyGate(x,y)
  ```

  * Do not need to try random search: the gradient is the direction of steepest increase. 
  * Evaluation of the derivative above only required three evaluations of the forward pass
    * Instead of hundreds for random search.
    * derivative provides the best tug one can hope for

* **Bigger step is not always better**: The math is for infinitesimal step sizes. For large steps, all bets are off. We hope that the function is smooth enough such that the step sizes we take work.

* **Hill-climbing analogy**: Like walking up a hill blindfolded: shuffle your feet in the direction of steepest ascent, but if you take a large step, you might fall off a cliff.

### Strategy #3: Analytic Gradient

* In the previous section, the derivative was computed by probing the ciruit's output value, independently for every input.
  * Yields **numerical gradient** 
  * Complexity is linear in the number of inputs (need to compute for each input)
* Better approach: **analytic gradient** from calculus
  * Idea: Compute gradient for small and simple expressions and then compose then with the **chain rule** 

> The analytic derivative requires no tweaking of the inputs. It can be derived using mathematics (calculus).

* Derivation from previous example of $f(x,y) = xy$
  $$
  \frac{\partial f(x,y)}{\partial x} = \frac{f(x+h,y) - f(x,y)}{h} = \frac{(x+h)y - xy}{h} = \frac{xy +hy -xy}{h} = \frac{hy}{y} = y
  $$
  (limits as h goes to zero omitted)

  * No tweaking required!

* Updated code using analytic gradient

  ```python
  x = -2
  y = 3
  out = forwardMultiplyGate(x,y)
  x_gradient = y
  y_gradient = x
  
  step_size = 0.01
  x += step_size * x_gradient
  y += step_size * y_gradient
  out_new = forwardMultiplyGate(x,y)
  ```

* **Progression:**

  * Strategy #1 (**random search**) required forwarding the circuit hundreds of times
  * Strategy #2 (**numerical gradient**) required forwarding twice the number of inputs (computation of numerator)
  * Strategy #3 (**analytic gradient**) requires forwarding a single time
  * Also strategies #1 and #2 only yield appoximate gradients, while strategy #3 gives the exact gradient.

* In practice implement analytic gradient, but verify by comparing it with the numerical gradient, which is easy, but expensive, to compute

### Recursive Case: Circuits with Multiple Gates

> A single extra multiplication will turn a single (useless gate) into a cog in the complex machine that is an entire neural network.

* Example circuit/function: $f(x,y,z) = (x+y)z$ 

![image-20201110062824975](figures/image-20201110062824975.png)

```python
def forwardMultiplyGate(a,b):
    return a * b

def forwardAddGate(a,b):
    return a + b

def forwardCircuit(x, y, z):
    q = forwardAddGate(x,y)
    f = forwardMultiplyGate(q,z)
    return f

x = -2
y = 5
z = -4

f = forwardCircuit(x,y,z)
```

* **Idea**: Ignore $x$ and $y$ and call the output of the add gate $q$, then we are back at the simple multiply gate with inputs $q$ and $z$.
  $$
  f(q,z) = qz \Rightarrow \frac{\partial f(q,z)}{\partial q} = z, \frac{\partial f(q,z)}{\partial z}=q
  $$

* Next, compute the gradient of $q(x,y) = x + y$ w.r.t. $x$ and $y$. 
  $$
  q(x,y) = x + y \Rightarrow \frac{\partial q(x,y)}{\partial x} = 1, \frac{\partial q(x,y)}{\partial y} = 1
  $$

### Backpropagation

* The **chain rule** tells us how to compute the gradient of the final output w.r.t. $x$ and $y$ given the gradient of $q$ w.r.t. $x$ and $y$ and the gradient of the final output w.r.t. $q$. 

  * Multiply the derivatives together!
    $$
    \frac{\partial f(q,z)}{\partial x} = \frac{\partial q(x,y)}{\partial x} \frac{\partial f(q,z)}{\partial q}
    $$
    

  ```python
  x = -2
  y = 5
  z = -4
  q = forwardAddGate(x,y)
  f = forwardMultiplyGate(q,z)
  
  # Gradient of the MULTIPLY gate w.r.t. its inputs
  derivative_f_wrt_z = q
  derivative_f_wrt_q = z
  
  # Gradient of the ADD gate w.r.t. its inputs
  derivative_q_wrt_x = 1.0
  derivative_q_wrt_y = 1.0
  
  # Chain rule
  derivative_f_wrt_x = derivative_q_wrt_x * derivative_f_wrt_q
  derivative_f_wrt_y = derivative_q_wrt_y * derivative_f_wrt_q
  
  # Final gradient
  gradient_f_wrt_xyz = [derivative_f_wrt_x,
                        derivative_f_wrt_y,
                        derivative_f_wrt_z,]
  
  # Let the inputs respond to the force/tug
  step_size = 0.01
  x += step_size * derivative_f_wrt_x
  y += step_size * derivative_f_wrt_y
  z += step_size * derivative_f_wrt_z
  
  q = forwardAddGate(x,y)
  z = forwardMultiplyGate(q,z)
  ```

* To increase the output value, imagine "pulling" up on the output

  * This induces a force on both $q$ and $z$
    * The circuit wants to increase $z$ since `derivative_f_wrt_z = +3` (i.e is positive)
      * The size of the derivative can be interpreted as the magnitude of the force
    * $q$ felt a stronger downward force: `derivative_f_wrt_q = -4`
      * $q$ wants to decrease with a force of 4
  * `q` is a function of `x,y`
    * **Crucial point**: The gradient of `q` was computed as negative, so the circuit wants `q` to decrease with a force of 4.
      * So if the `+` gate wants to contribute to making the final input value larger, it needs to listen to the gradient signal coming from the top. 
      * In this case, it needs to apply tugs on `x,y` opposite what it would normally apply with a force of 4. 
      * The multiplication by `-4` in the chain rule accomplishes this. 
        * Instead of applying a positive force of `+1` on both `x` and `y` (the local derivatives), the full circuit's gradient on both `x` and `y` becomes `1 x -4 = -4`. 
        * This makes sense since the circuit wants both `x` and `y` to get smaller because it will make `q` smaller, which in turn makes `f` larger

* **Recap**

  * For a since gate, use calculus to derive the analytic gradient. Interpret the gradient as a force, or tug, on the inputs that pulls them in a direction which would make the gate's output higher
  * For multiple gates, every gate is hanging out by itself completely unaware of the circuit it is embedded in.
    * The **only** difference is that something can pull on this gate from above: the gradient of the final circuit output value w.r.t. the output of the gate.
    * The gate simply takes this force and multiplies it to all the forces it computed for its input before (this is the chain rule)
  * If a gate experiences a strong positive pull from above, it will also pull harder on its own inputs, scaled by the force it is experiencing from above
  * If it experiences a negative tug, this means the circuit wants its value to decrease, not increase, so it will flip the force of the pull on its inputs to make its own output value smaller

> A nice picture to have in mind is that as we pull on the circuit's output value at the end, this induces pulls downward through the entire circuit, all hte way down to the inputs.

* Repeat: the **only** difference between the case of a single gate and the multiple interacting gates that compute arbitrarily complex expressions is this additional multiply operation that now happens in each gate.

### Patterns in the "backward" flow

* Look at the previous circuit with values filled in

  ![image-20201110070136533](figures/image-20201110070136533.png)

* The first circuit is the raw values of the forward pass. The second circuit are the gradients.
  
  * The gradient always starts off with a value of `+1` at the end to start off the chain. This is the (default) pull on the circuit to have its value increased.
* Patterns:
  * `+` gate simply passes the gradient to all its inputs
  * `max(x,y)` is `+1` for the larger of `x` and `y`, and `0` for the other
    * Acts like a switch: takes the gradient from above and "routes" it to the input that had a higher value during the forward pass. 

#### Numerical Gradient Check

```python
x = -2
y = 5
z = -4

# numerical gradient check
h = 0.0001
dx = (forwardCircuit(x+h,y,z) - forwardCircuit(x,y,z)) / h # -4
dy = (forwardCircuit(x,y+h,z) - forwardCircuit(x,y,z)) / h # -4
dz = (forwardCircuit(x,y,z+h) - forwardCircuit(x,y,z)) / h # 3
```

## Example: Single Neuron

* Example in this section:
  $$
  f(x,y,a,b,c) = \sigma(ax + by + c)
  $$
  where $\sigma$ is the sigmoid function
  $$
  \sigma(x) = \frac{1}{1+e^{-x}}
  $$

  * This is a "squashing function" since it takes any input and squashes it to be between 0 and 1. 

* The derivative of $\sigma$ is
  $$
  \frac{\partial \sigma(x)}{\partial x} = \sigma(x)(1-\sigma(x))
  $$

* This is all we need to use this gate! 

  * Know how to take an input and forward it through the sigmoid gate
  * We have an expression for the gradient w.r.t. its input so we can **backprop** through it.

* Note the sigmoid is itself composed of more **atomic** functions: exponentiation, addition, and division. Treat it as a single unit moving forward.

* Every wire in our circuits has two numbers associated with it

  1. the value it carries during the forward pass
  2. the gradient (i.e. the **pull**) that flows back through it in the backward pass
  3. Package this up:

  ```python
  # every Unit corresponds to a wire in the diagrams
  class Unit:
      def __init__(self, value, grad):
          self.value = value
          self.grad = grad
  ```

* Next build the multiply gate

  ```python
  class MultiplyGate:
      def __init__(self):
          self.u0 = None
          self.u1 = None
          self.utop = None
          
      def forward(self, u0, u1):
          self.u0 = u0
          self.u1 = u1
          self.utop = Unit(u0.value * u1.value, 0.0)
          return self.utop
      
      def backward(self):
          self.u0.grad += self.u1.value * self.utop.grad
          self.u1.grad += self.u0.value * self.utop.grad
  ```
  
  * The multiply gate takes two inputs that each hold a value and creates a unit that stores its output.
  * The gradient is initialized to zero. 
* In the backward pass we get the gradient from the output unit that we generated during the forward pass and multiply it with the local gradient for this gate (the chain rule). 
  * Use `+=` to add onto the gradient in the `backward` function. This allows us to possibly use the output of one gate multiple times (think of wire branching), since it turns out that the gradients from these different branches just add up when computing the final gradient w.r.t. the circuit output. 
  
* Add Gate:

  ```python
  class AddGate:
      def __init__(self):
          self.u0 = None
          self.u1 = None
          self.utop = None
          
      def forward(self, u0, u1):
          self.u0 = u0
          self.u1 = u1
          self.utop = Unit(u0.value + u1.value, 0.0)
          return self.utop
      
      def backward(self):
          self.u0.grad += 1.0 * self.utop.grad
          self.u1.grad += 1.0 * self.utop.grad
  ```

* Sigmoid Gate:

  ```python
  import math
  
  class SigmoidGate:
      def __init__(self):
          self.u0 = None
          self.utop = None
          
      def forward(self, u0):
          self.u0 = u0
          self.utop = Unit(self.sig(u0.value), 0.0)
          return self.utop
      
      def backward(self):
          s = self.sig(self.u0.value)
          self.u0.grad += (s*(1-s)) * self.utop.grad
          
      def sig(self, x):
          return 1.0 / (1.0 + math.exp(-x))
  ```

* Create units and gates

  ```python
  # Create input units
  a = Unit(1.0, 0.0)
  b = Unit(2.0, 0.0)
  c = Unit(-3.0, 0.0)
  x = Unit(-1.0, 0.0)
  y = Unit(3.0, 0.0)
  
  # Create the gates
  mulg0 = MultiplyGate()
  mulg1 = MultiplyGate()
  addg0 = AddGate()
  addg1 = AddGate()
  sg0 = SigmoidGate()
  ```

* Link them together

  ```python
  def forwardNeuron():
      ax = mulg0.forward(a, x)
      by = mulg1.forward(b, y)
      axpby = addg0.forward(ax, by)
      axpbypc = addg1.forward(axpby, c)
      s = sg0.forward(axpbypc)
      return s
    
  s = forwardNeuron()
  ```

* Backprogation

  ```python
  s.grad = 1.0
  sg0.backward()
  addg1.backward()
  addg0.backward()
  mulg1.backward()
  mulg0.backward()
  ```

  * Note we initialize the gradient at the outptu to be 1.0
    * Tugging on the entire network with a force of `+1` to increase the output value.

* Update parameters

  ```python
  step_size = 0.01
  a.value += step_size * a.grad
  b.value += step_size * b.grad
  c.value += step_size * c.grad
  x.value += step_size * x.grad
  y.value += step_size * y.grad
  ```

* Verify backpropagation with numerical gradient computation

  ```python
  def forwardCircuitFast(a,b,c,x,y):
      return 1/(1+math.exp(-(a*x + b*y + c)))
    
  a = 1
  b = 2
  c = -3
  x = -1
  y = 3
  h = 0.0001
  
  a_grad = (forwardCircuitFast(a+h,b,c,x,y) - forwardCircuitFast(a,b,c,x,y)) / h
  b_grad = (forwardCircuitFast(a,b+h,c,x,y) - forwardCircuitFast(a,b,c,x,y)) / h
  c_grad = (forwardCircuitFast(a,b,c+h,x,y) - forwardCircuitFast(a,b,c,x,y)) / h
  x_grad = (forwardCircuitFast(a,b,c,x+h,y) - forwardCircuitFast(a,b,c,x,y)) / h
  y_grad = (forwardCircuitFast(a,b,c,x,y+h) - forwardCircuitFast(a,b,c,x,y)) / h
  ```

  