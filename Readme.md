# Cartoonification Task
## Description 
It's required to develope a model for an entertainment app in which users will provide pictures for you to apply a cartoon effect on (Make them in the style of cartoons).
Your ideal input would be a frontal image of the user and you should provide the outputted image with the style applied.

for such a problem we will be using a specific type of deep learning Networks: GANs (Generative Adversarial Networks).

# Introducion to GANs
Generative adversarial networks (GANs) are an exciting recent innovation in machine learning. GANs are generative models: they create new data instances that resemble your training data. For example, GANs can create images that look like photographs of human faces, even though the faces don't belong to any real person.

GANs achieve this level of realism by pairing a generator, which learns to produce the target output, with a discriminator, which learns to distinguish true data from the output of the generator. The generator tries to fool the discriminator, and the discriminator tries to keep from being fooled.
for example if we want to build an animal idenrifier, A generative model could generate new photos of animals that look like real animals, while a discriminative model could tell a dog from a cat. GANs are just one kind of generative model.

More formally, given a set of data instances X and a set of labels Y:

- Generative models capture the joint probability p(X, Y), or just p(X) if there are no labels.
- Discriminative models capture the conditional probability p(Y | X).

Note that this is a very general definition. There are many kinds of generative model. GANs are just one kind of generative model.
