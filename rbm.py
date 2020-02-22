# +
from util import *
import tqdm
import os

tqdm.tqdm = tqdm.tqdm_notebook
import matplotlib.pyplot as plt


# -

class RestrictedBoltzmannMachine:
    """
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    """

    def __init__(
        self,
        ndim_visible,
        ndim_hidden,
        is_bottom=False,
        image_size=[28, 28],
        is_top=False,
        n_labels=10,
        batch_size=10,
    ):

        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end. 
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """

        self.ndim_visible = ndim_visible

        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom

        if is_bottom:
            self.image_size = image_size

        self.is_top = is_top

        if is_top:
            self.n_labels = 10

        self.batch_size = batch_size

        self.delta_bias_v = 0

        self.delta_weight_vh = 0

        self.delta_bias_h = 0

        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))

        self.weight_vh = np.random.normal(
            loc=0.0, scale=0.01, size=(self.ndim_visible, self.ndim_hidden)
        )

        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))

        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0

        self.weight_v_to_h = None

        self.weight_h_to_v = None

        self.learning_rate = 0.01

        self.momentum = 0.7

        self.print_period = 50

        self.rf = {  # receptive-fields. Only applicable when visible layer is input data
            "period": 5000,  # iteration period to visualize
            "grid": [5, 5],  # size of the grid
            "ids": np.random.randint(
                0, self.ndim_hidden, 25
            ),  # pick some random hidden units
        }

        return

    def cd1(self, visible_trainset, n_iterations=10000, binary_vis=True, print_img=False, img_dir='', return_loss = False):

        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
          binary_vis (boolean): determines wheater the input data is transfered into binary data or not
          print_img (boolean): if True, images are logged during training
          img_dir (str): directory in which rfs should be saved
          return_loss (boolean): If yes, list of loss after ever self.print_period is returned
        """

        print("learning CD1")

    
        recon_losses = []
        n_samples = visible_trainset.shape[0]
        if binary_vis == True: 
            visible_trainset = sample_binary(visible_trainset)
        num_it_per_epoch = int((n_samples / self.batch_size))
        for it in tqdm.tqdm(range(n_iterations)):
            start_idx_batch = (it % num_it_per_epoch) * self.batch_size
            end_idx_batch = ((it % num_it_per_epoch) + 1) * self.batch_size
            v_0 = visible_trainset[start_idx_batch:end_idx_batch,:]
            p_h, h_0 = self.get_h_given_v(v_0)
            p_v_1, v_1 = self.get_v_given_h(h_0)  # change back to h_0
            p_h_1, h_1 = self.get_h_given_v(v_1)  # change back to v_0

            self.update_params(v_0, h_0, p_v_1, p_h_1)

            if it % self.rf["period"] == 0 and self.is_bottom:

                viz_rf(
                    weights=self.weight_vh[:, self.rf["ids"]].reshape(
                        (self.image_size[0], self.image_size[1], -1)
                    ),
                    it=it,
                    grid=self.rf["grid"],
                    dir=img_dir
                )

                if print_img is True: 
                    fig, ax = plt.subplots(1, 4, figsize=(20, 20))
                    ax[0].imshow(v_0[0].reshape((28, 28)))
                    ax[0].set_title('sam. 1')
                    ax[1].imshow(v_1[0].reshape((28, 28)))
                    ax[1].set_title('recon. sam. 1')
                    ax[2].imshow(v_0[1].reshape((28, 28)))
                    ax[2].set_title('sam. 2')
                    ax[3].imshow(v_1[1].reshape((28, 28)))
                    ax[3].set_title('recon. sam. 2')
                    plt.show()
            
            #Print loss
          
            if it % self.print_period == 0:    
                h_entire_data_set = self.get_h_given_v(visible_trainset)[1]
                if binary_vis == 'True':
                    v_out = self.get_v_given_h(h_entire_data_set)[1]
                else:
                    v_out = self.get_v_given_h(h_entire_data_set)[0]
                recon_loss = np.linalg.norm(np.mean((visible_trainset - v_out), axis=1))
                print(  "iteration=%7d recon_loss=%4.4f"% (it, recon_loss ))
                if return_loss:
                    recon_losses.append(recon_loss)
        if return_loss:
            return recon_losses
        else: 
            return 

    def update_params(self, v_0, h_0, v_k, h_k):

        """Update the weight and bias parameters.

        You could also add weight decay and momentum for weight updates.

        Args:
           v_0: activities or probabilities of visible layer (data to the rbm)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer
           h_k: activities or probabilities of hidden layer
           all args have shape (size of mini-batch, size of respective layer)
        """

        batch_size = h_0.shape[0]

        self.delta_bias_v += self.learning_rate * np.mean(v_0 - v_k, axis=0)
        self.delta_weight_vh = self.learning_rate * (
            np.dot(v_0.T, h_0)  - np.dot(v_k.T, h_k)
        )  # ToDo: Test if with / batch_size or without
        self.delta_bias_h += self.learning_rate * np.mean(h_0 - h_k, axis=0)

        self.bias_v += self.delta_bias_v
        self.weight_vh += self.delta_weight_vh
        self.bias_h += self.delta_bias_h

        return

    def get_h_given_v(self, visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v) 

        Uses undirected weight "weight_vh" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_vh is not None

        n_samples = visible_minibatch.shape[0]
        p_h = sigmoid(
            self.bias_h + np.dot(visible_minibatch, self.weight_vh)
        )  # for entire mini-batch
        h = sample_binary(p_h)

        return p_h, h

    def get_v_given_h(self, hidden_minibatch):

        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_vh is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            support = self.bias_v + np.dot(hidden_minibatch, self.weight_vh.T)  # get probabilities
            pv_label = softmax(support[:, :self.n_labels]) # split to sample different for labels and img
            pv_img = sigmoid(support[:, self.n_labels:])
            p_v = np.concatenate((pv_label,pv_img),axis=1)
            v_img = sample_binary(pv_img)  # Flip to zero or one
            v_label = sample_categorical(pv_label)  # Get one '1' for entire labels of one image
            v = np.concatenate((v_label,v_img),axis=1) # concatenate again 

        else:

            p_v = sigmoid(self.bias_v + np.dot(hidden_minibatch, self.weight_vh.T))
            v = sample_binary(p_v)

        return p_v, v

    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """

    def untwine_weights(self):

        self.weight_v_to_h = np.copy(self.weight_vh)
        self.weight_h_to_v = np.copy(np.transpose(self.weight_vh))
        self.weight_vh = None

    def get_h_given_v_dir(self, visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_v_to_h is not None

        n_samples = visible_minibatch.shape[0]

        p_h = sigmoid(
            self.bias_h + np.dot(visible_minibatch, self.weight_v_to_h) #same computation as the function 'get_h_given_v' but with directed connections 
        )  # for entire mini-batch
        h = sample_binary(p_h)

        return p_h, h

    def get_v_given_h_dir(self, hidden_minibatch):

        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_h_to_v is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            raise Exception('The top layer does not habe direct connections in a DBN. This function can not be called with top layer')

        else:

            p_v = sigmoid(self.bias_v + np.dot(hidden_minibatch, self.weight_h_to_v))
            v = sample_binary(p_v)

        return p_v, v

    def update_generate_params(self, inps, trgs, p_preds):

        """Update generative weight "weight_h_to_v" and bias "bias_v"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_h_to_v += self.learning_rate * np.dot(inps.T, (trgs-p_preds))
        self.delta_bias_v += self.learning_rate * np.mean(trgs - p_preds, axis=0)

        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v += self.delta_bias_v

        return

    def update_recognize_params(self, inps, trgs, p_preds):

        """Update recognition weight "weight_v_to_h" and bias "bias_h"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """
        #update weights according to lab slides

        self.delta_weight_v_to_h += self.learning_rate * np.dot(inps.T, (trgs-p_preds))
        self.delta_bias_h += self.learning_rate * np.mean(trgs - p_preds, axis=0)

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h += self.delta_bias_h

        return
