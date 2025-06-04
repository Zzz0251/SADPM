
from denoising_diffusion_pytorch import Unet2, GaussianDiffusion, Trainer, Unet
from PIL import Image
import numpy as np
import torch
import os




dataset_name = 'Datasetxxx_synapse'
folder_training=fr'./SADPM/data/{dataset_name}/train/'


dataset = 'synapse'


folder_result=fr'./TrainingResults/{dataset}/'


model = Unet2(
    dim = 32,                                
    dim_mults = (2, 4, 4, 8, 16, 16, 16),   
    channels = 1,                           
    learned_sinusoidal_dim = 16            
).cuda()                                   


diffusion = GaussianDiffusion(
    model,           
    image_size = ,              
    timesteps = ,              

    sampling_timesteps = ,      

 
    ddim_sampling_eta = 1.,       

 
    loss_type = 'l2',              
    loss_type_add_l1 = False,      
    loss_type_add_l2 = False,      

    beta_schedule = 'sigmoid',      


    p2_loss_weight_gamma = 1.,     
    p2_loss_weight_k = 1,          
    add_posterior_noise = True,    
    lamda1 = 0.95,                 
    lamda2 = 1,                    
    posterior_D = 3000,            
    posterior_folder = folder_training,      
    posterior_folder_result = folder_result, 
).cuda()                           

trainer = Trainer(
    diffusion,                      
    folder_training,                
    train_batch_size = 2,          
    train_lr = 1e-4,             
    min_training_lr = 6e-5,       
    save_and_sample_every = ,  
    train_num_steps = ,       

    gradient_accumulate_every = 1, 
    optimizer_type = 'Adam',      

    training_schedual = '',  #  consistant  linear_decrease cosine_annealing
    ema_decay = 0.995,            
    amp = False,                  
    num_samples = 4,              
    augment_horizontal_flip = True,  
    results_folder_absolute = folder_result,  
    plot_folder = '', 
    if_posterior_predict = True,   
    lamda1 = 0.95,                
    lamda2 = 1,                   
    posterior_D = 3000,           
)


# trainer.load(0)
trainer.train()

# trainer.predict(fr'../Testset/..', fr'../Predicts/..', False, True)
