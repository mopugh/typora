# Blog Ideas

## Unorganized

- Why Gaussian? Does Gaussian make VAE continuous? What are the effects of using other distributions (need to compute KL? Consider Laplace distribution?) What happens if latent space is larger? 
- What happens if we use a ReLU output for a VAE and regular variance rather than log of the variance and a non-activated output?
- Compare VAE in tf vs pytorch
- Use of higher dimensional latent space? Use of sparse priors and l1 penalty (rather than KL? or how are they related?)
- What happens if you switch order or optimization: generator before discriminator
- What happens if we had two discriminators? One WGAN and one NS-GAN? (Multi-agent systems?)
- Generally, what happens if we switch distributions in GANs
- Can we do bandits with a neural network for each arm? Density estimation plus reparameterization trick?
- Why isn't reparameterization always possible? Transform a uniform random variable and use neural network as function approximator for inverse CDF?
- Is generalized policy iteration a game? Can you use min-max methods? Compare Jacobi vs Gauss-Siedel policy iteration (notes chapter 4 Sutton)
- Adversarial dropout? Take norm of gradient components and sample, i.e. larger norms are selected more frequently?
- Adversarial Autoencoders? Regularized?
- GAN bandit algorithms
- Backpropagation through the VOID + WGAN without Lipschitz condition?
- Different metric for VAE (mutual information between input and output of encoder rather than KL with Gaussian) Use Backpropagation through the VOID for non Gaussian?
- Use hyperparameter optimization on a basic GAN? (See inference.vc and David D.) Use of package "Higher"
- GAN feature matching and kernels?
- Can we use second order optimization (Newton sketch?) for better GAN performance?
- Compare differentiable games and numerics of gans
- Vector field design: http://brickisland.net/DDGSpring2020/
- Work through examples of graph neural networks
- Can we use learning a graph from data to get a structure to use for VAE?
- What happens if we replace averages with maximums in loss functions?
- Use David D. gradient estimator with VAE? (Does not work well; see Kingma/Welling VAE review section 2.9.1)
- Use of earth-movers distance rather than principle of maximum likelihood
- search sinkhorn in zotero
- Combine Stanford paper on gradient from data with backprop through the void
- Compare VAEs and normalized flows
  - Look at diffusion VAEs
  - Hyperbolic VAEs and normalizing flows
  - Continuous Hierarchical Representations with Poincaré Variational Auto-Encoders