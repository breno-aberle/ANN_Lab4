from util import *
from rbm import RestrictedBoltzmannMachine
from datetime import date, datetime
import tqdm
tqdm.tqdm = tqdm.tqdm_notebook
import os

class DeepBeliefNet():    

    ''' 
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis] 
                               `-> [lbl] 
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    '''
    
    def __init__(self, sizes, image_size, n_labels, batch_size):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {
            
            'vis--hid' : RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                    is_bottom=True, image_size=image_size, batch_size=batch_size),
            
            'hid--pen' : RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"], batch_size=batch_size),
            
            'pen+lbl--top' : RestrictedBoltzmannMachine(ndim_visible=(sizes["pen"]+sizes["lbl"]), ndim_hidden=sizes["top"],
                                                        is_top=True, n_labels=n_labels, batch_size=batch_size)
        }
        self.n_labels = n_labels

        self.sizes = sizes

        self.image_size = image_size

        self.batch_size = batch_size
        
        self.n_gibbs_recog = 15
        
        self.n_gibbs_gener = 200
        
        self.n_gibbs_wakesleep = 5

        self.print_period = 2000
        
        return

    def recognize(self,true_img,true_lbl):

        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """
        
        n_samples = true_img.shape[0]
        
        vis = true_img # visible layer gets the image data
        
        lbl = np.ones(true_lbl.shape)/10. # Initilize labels
        
        # drive the network bottom to top
        #ToDo: Calculate h with (get_h_given_v_dir) with rbm vis-hid
        p_h_vis, h_vis = self.rbm_stack['vis--hid'].get_h_given_v_dir(vis) 
        #ToDo: Calculate h with (get_h_given_v_dir) with rbm hid-pen
        p_h_hid, h_hid = self.rbm_stack['hid--pen'].get_h_given_v_dir(vis)
        
        for _ in range(self.n_gibbs_recog):
            #ToDo: run alternating gibbs sampling with pen+lbs--top with cd1
            self.rbm_stack['hid--pen'].cd1(visible_trainset = h_hid)
            #pass

        #ToDo: read out labels from run (take first 10 columns) to predicted_lbl
        predicted_lbl = np.zeros(true_lbl.shape)
            
        print ("accuracy = %.2f%%"%(100.*np.mean(np.argmax(predicted_lbl,axis=1)==np.argmax(true_lbl,axis=1))))
        
        return

    def generate(self,true_lbl,name):
        
        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """
        
        n_sample = true_lbl.shape[0]
        
        records = []        
        fig,ax = plt.subplots(1,1,figsize=(3,3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([]); ax.set_yticks([])

        lbl = true_lbl

        # [TODO TASK 4.2] fix the label in the label layer and run alternating Gibbs sampling in the top RBM. From the top RBM, drive the network \ 
        # top to the bottom visible layer (replace 'vis' from random to your generated visible layer).
            
        for _ in range(self.n_gibbs_gener):

            vis = np.random.rand(n_sample,self.sizes["vis"])
            
            records.append( [ ax.imshow(vis.reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None) ] )
            
        anim = stitch_video(fig,records).save("%s.generate%d.mp4"%(name,np.argmax(true_lbl)))            
            
        return

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations, train_top_layer=True, img_dir=''):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack. 
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
          train_top_layer (boolean): If False only bottom and middle layer are computed
          img_dir (str): directory in which rfs should be saved
        """

        try :
            #Load RBM
            self.loadfromfile_rbm(loc="trained_rbm",name="vis--hid")
            self.loadfromfile_rbm(loc="trained_rbm",name="hid--pen")
            self.loadfromfile_rbm(loc="trained_rbm",name="pen+lbl--top")      
            
            # Untwine Weight for first two layers 
            self.rbm_stack["vis--hid"].untwine_weights()  
            self.rbm_stack["hid--pen"].untwine_weights()

        except IOError :
            print ("training vis--hid")
            img_dir_vis = os.path.join(img_dir, 'vis--hid')
            os.makedirs(img_dir_vis)
            # Gibbs sampling k=1 for vis to hid
            self.rbm_stack["vis--hid"].cd1(visible_trainset=vis_trainset, n_iterations=n_iterations, img_dir=img_dir_vis)
            self.savetofile_rbm(loc="trained_rbm",name="vis--hid")

            print ("training hid--pen")
            self.rbm_stack["vis--hid"].untwine_weights()
            hid_trainset = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis_trainset)[1]
            img_dir_hid = os.path.join(img_dir, 'hid--pen')
            os.makedirs(img_dir_hid)
            self.rbm_stack["hid--pen"].cd1(visible_trainset=hid_trainset, n_iterations=n_iterations, img_dir=img_dir_hid)          
            self.savetofile_rbm(loc="trained_rbm",name="hid--pen") 
            
            if train_top_layer:
                print ("training pen+lbl--top")
                self.rbm_stack["hid--pen"].untwine_weights()
                pen_trainset = self.rbm_stack["hid--pen"].get_h_given_v_dir(hid_trainset)[1] 
                pen_trainset = np.concatenate((lbl_trainset,pen_trainset),axis=1)
                img_dir_pen = os.path.join(img_dir, 'pen+lbl--top')
                os.makedirs(img_dir_pen)
                self.rbm_stack["pen+lbl--top"].cd1(visible_trainset=pen_trainset, n_iterations=n_iterations, img_dir=img_dir_pen)          
                self.savetofile_rbm(loc="trained_rbm",name="pen+lbl--top")            

        return    

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Wake-sleep method for learning all the parameters of network. 
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """
        
        print ("\ntraining wake-sleep..")

        try :
            
            self.loadfromfile_dbn(loc="trained_dbn",name="vis--hid")
            self.loadfromfile_dbn(loc="trained_dbn",name="hid--pen")
            self.loadfromfile_rbm(loc="trained_dbn",name="pen+lbl--top")
            
        except IOError :            

            self.n_samples = vis_trainset.shape[0]
            num_it_per_epoch = int((self.n_samples / self.batch_size)) 
            for it in tqdm.tqdm(range(n_iterations)): 
                start_batch = (it % num_it_per_epoch) * self.batch_size
                end_batch = ((it % num_it_per_epoch) + 1) * self.batch_size
                vis = vis_trainset[start_batch:end_batch,:]
                lbl = lbl_trainset[start_batch:end_batch,:]
                #### vis -> hid -> pen (+lbl) -> top ####     

                # wake-phase (according to slide page 24)
                p_hid, hid = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis)
                p_pen, pen = self.rbm_stack["hid--pen"].get_h_given_v_dir(hid)

                #alternating Gibbs sampling (according to page 25)
                pen_lbl_0 = np.concatenate((lbl,pen),axis=1)
                p_top_0, top_0 = self.rbm_stack["pen+lbl--top"].get_h_given_v(pen_lbl_0)
                pen_lbl_k = np.copy(pen_lbl_0)
                for _ in range(self.n_gibbs_wakesleep):
                  p_top_k, top_k = self.rbm_stack["pen+lbl--top"].get_h_given_v(pen_lbl_k)
                  p_pen_lbl_k, pen_lbl_k = self.rbm_stack["pen+lbl--top"].get_v_given_h(top_k)  # do we need to loop? # Do we need to end up at 500+10 units

                # sleep phase (page 26)
                p_gen_hid, gen_hid = self.rbm_stack["hid--pen"].get_v_given_h_dir(pen_lbl_k[:,self.n_labels:])  #change input to generated one
                p_v_out, v_out =  self.rbm_stack["vis--hid"].get_v_given_h_dir(gen_hid)

                # Update generative parameters based on results from wake phase (page 28)
                self.rbm_stack["vis--hid"].update_generate_params(hid, vis, p_v_out) #Question: Do we need to calculate pv_out again?
                self.rbm_stack["hid--pen"].update_generate_params(pen, hid,  p_gen_hid)
                
                # Update top rbb with update_params (page 29)
                self.rbm_stack["pen+lbl--top"].update_params(pen_lbl_0, top_0, pen_lbl_k, top_k)
                
                #pdate recognize parameters (page 30)
                self.rbm_stack["hid--pen"].update_recognize_params(gen_hid, pen_lbl_k[:,self.n_labels:],  p_pen) #Question: Do we need to calculate p_pen again? 
                self.rbm_stack["vis--hid"].update_recognize_params(v_out, gen_hid,  p_hid)
                        
            self.savetofile_dbn(loc="trained_dbn",name="vis--hid")
            self.savetofile_dbn(loc="trained_dbn",name="hid--pen")
            self.savetofile_rbm(loc="trained_dbn",name="pen+lbl--top")            

        return

    
    def loadfromfile_rbm(self,loc,name):
        
        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy"%(loc,name))
        self.rbm_stack[name].bias_v    = np.load("%s/rbm.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h    = np.load("%s/rbm.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_rbm(self,loc,name):
        
        np.save("%s/rbm.%s.weight_vh"%(loc,name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v"%(loc,name),    self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h"%(loc,name),    self.rbm_stack[name].bias_h)
        return
    
    def loadfromfile_dbn(self,loc,name):
        
        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy"%(loc,name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy"%(loc,name))
        self.rbm_stack[name].bias_v        = np.load("%s/dbn.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h        = np.load("%s/dbn.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_dbn(self,loc,name):
        
        np.save("%s/dbn.%s.weight_v_to_h"%(loc,name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v"%(loc,name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v"%(loc,name),        self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h"%(loc,name),        self.rbm_stack[name].bias_h)
        return

