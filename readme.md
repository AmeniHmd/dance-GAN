To Run the model in evaluation/inference mode:


python DanceDemo.py --GEN_TYPE=value
where GEN_TYPE is one of the following methods:
    # NEAREST = 1
    # VANILLA_NN_SKE = 2
    # VANILLA_NN_Image = 3
    # GAN = 4
    
    
To train GenVannilaNN using skeleton data:
python GenVanillaNN.py --optSkeOrImage=1 --n_epochs=2000

To train GenVanillaNN using a skeleton represented as an image:
python GenVanillaNN.py --optSkeOrImage=2 --n_epochs=2000

To train GenGAN using a skeleton represented as an image:
python GenGAN.py --n_epochs=2000


Section 1: GenVanillaNN from Skeleton Data
- Having as input a vector of 26 data from the skeletpn, makes it the input noise vector to the generation model( in terms of adversarial generation models)
- The fact of having a vector of [26, 1, 1] as size, means we needs to upscale or upsample the the width and height from 1x1 to 64x64, while producing more channels with informtive data to build the final image.
- A model where we stack many ConvTranspose2d layers taking into consideration the constraint of going from 1x1 -> 64x64 using the following formula:
output_size = (input_size -1)xstride - 2xpadding + kernel_size + outputpadding

- We add a  BatchNorm2d layer after each hidden convolutional layer helps stabilize the learning process by normalizing the output of each layer. We set affine to true: means that the Batch Normalization layer will have learnable weights (scale) and biases (shift) -> This allows the model to retain the capacity to represent the original distribution of inputs if needed.

-  we add a ReLU as activation to introduce non-linearity.

- we only add Tanh as activation to constrain the output values between [-1,1]: allows for smooth color transitions and gives the generator better control over shading and contrast |  improves gradient flow during backpropagation, helping to avoid issues like vanishing gradients | Normalizations leads to a more stable train.

Section 2: GenVanillaNN from Skeleton Image
- Having as input a vector of [3, 64, 64] image data from the skeleton, makes it a complex input as the WxH is already equal to the output pixels number. In this case we need to encode/decode or contract/expand sub-models.
- the expanding part is a stack of convolutional layers( Conv2d), where we go from [3,64,64] to [256,1,1] using the following formula: 
output_size = ((input_dim - kernel_size + 2xpadding)/stride) + 1

- The expanding block is similar to the model in the genvanilla with skeleton data, but we used InstanceNorm2d. It was originally proposed to improve style transfer tasks by normalizing feature statistics (mean and variance) per instance (i.e., per image) rather than across the entire batch.
After Being published in 2016, several GAN-based architectures adopted this technique, such as Pix2Pix and CycleGAN, for tasks involving image-to-image translation and style transfer, where preserving specific features and styles per instance was crucial.

General information about training:
- We use MSELoss as we compare the difference between two images.
- The optimizer is SGD with a learning rate lr=0.001( we can use a smaller value to make the gradient steps smaller and converge to the minimum more stable).

Section 3: GenGAN from Skeleton Image
- the GENERATOR in this model will be GenVanillaNN from Skeleton Image.
- The discriminator is a classification task detecting whether is given image is real or fake, while the generator will make its best to fool the deiscriminator to see the fake generated images as real images.

- the problem here is that a classification task is easier that an image generation task -> our challenge is to keep both models improve at the same pace.
- the basic order is applied: Conv2d, BatchNorm2d, LeakyReLU for hidden layers. And a Conv2d convolution followed by a sigmoid( binary classification) for the last layer.

-> we used LeakyReLU instead of ReLU, as LeakyReLU allows the discriminator to:

Maintain non-zero gradients for negative inputs, preventing "dying" neurons || Provide more stable gradients for training, helping to balance the adversarial game || Serve as a standard activation in GAN discriminators due to its effectiveness in practice.

- We applied the spectral normalization from the paper SN-GAN: first becausse it helps improve stability and avoid vanishing gradient, such as mode collapse. 

General information about training:

- We have two sub-training parts:

- The first is to improve the descrimination:
* We use BCELoss as it is a binary classification. 
The discriminator loss is both for real_classification_loss and fake_classification_loss averaged. 
For the sake of synchronization between the generator and the discriminator we will use few tricks in the loss function:
1. First by adding a gradient penalty which can regularize the discriminator by penalizing large gradients, which prevents it from learning too quickly and becoming too strong => inspired by Wasserstein GANs (WGAN-GP).
2. By using label smoothing (slightly lowering the target labels from 1.0 to, for example, 0.9 for real images) can help reduce overconfidence, making the GAN training more stable. As well as for fake images( 0.1 instead of 0).
3. By Reducing the discriminator’s update frequency relative to the generator. For instance, update the discriminator every 2nd or generator step.
4. By using smaller batches compared to the generator batch size( 128 versus 32)

* Adam optimizer as it is widely used in generative models because it: Adapts learning rates for faster, stable convergence.
Handles noisy gradients well, stabilizing adversarial training || Is robust to small batch sizes, beneficial for memory-intensive models || Has shown consistent empirical success in various generative architectures.


- The second is to improve the generator:
* We have two losses:
1. BCELoss feedback: important for the model the know if it is folloing the discriminator.
2. MSELoss: for matching real images in pixel-space. This may be ricky as it can make the generator prioritize pixel-wise similarity over fooling the discriminator.

* We used as well Adam optimizer.











