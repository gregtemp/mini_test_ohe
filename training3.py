import torch
import numpy as np
from model_create2 import BTCVAE
from custom_dataloader import CustomDataset
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import datetime
import os
import preprocess

### analyze dataset separately and save preproc_state before using this script
preproc = preprocess.Preprocessor(in_data_size=195, n_bins=15)
preproc.load_state('preproc_state.pkl')


architecture_file = 'architecture2.txt'  # Replace this with the path to your text file
dataset_filename = "operator-presets2-norm_mod.csv"
batch_size = 150
# single opt
learning_rate = 0.005
# Set learning rates for the KLD and MSE losses
kld_learning_rate = 0.00001
mse_learning_rate = 0.00001

num_epochs = 300
loss_scaler = 1000.

start_seed = 42
momentum = 0.90

for seed in range(1):
    seed += start_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    nan_encountered = False
    loss_log =[]
    # Instantiate the model and optimizer
    model = BTCVAE(architecture_file, alpha=0.1, beta=10.0).cuda()
    print(model)

    # Load the dataset
    data = pd.read_csv(dataset_filename)
    data = data.iloc[:, :-1]
    data.fillna(0., inplace=True)
    dataset = CustomDataset(data, seed)
    eval_dataset = TensorDataset(torch.tensor(data.values, dtype=torch.float32))

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    eval_data_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    # single optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # double optimizer
    # Create parameter groups for the encoder and decoder
    # encoder_params = list(model.encoder.parameters())
    # decoder_params = list(model.decoder.parameters())
    # Create separate optimizers for the encoder and decoder
    #kld_optimizer = optim.SGD(encoder_params, lr=kld_learning_rate, momentum=momentum)
    #mse_optimizer = optim.SGD(decoder_params, lr=mse_learning_rate, momentum=momentum)

    ###

    #print(f" test print out data size : {preproc.out_data_size}")

    # Training loop
    for epoch in range(num_epochs):
        batch_idx = 0
        for batch in data_loader:
            batch_idx+=1
            x = batch
            x - x.float()
            x = x.cuda()
            print(f"x type {x.dtype}")
            print(x.dtype)
            print(model.encoder[0].weight.dtype)
            x_recon, mu, logvar = model(x)
            loss = model.loss_function(x, x_recon, mu, logvar, preproc.ohe_check, preproc.ohe_edges, preproc.nunique)
            # single opt
            optimizer.zero_grad()
            # double opt
            #kld_optimizer.zero_grad()
            #mse_optimizer.zero_grad()

            loss.backward()
            # single opt
            optimizer.step()
            #kld_optimizer.step()
            #mse_optimizer.step()
            
            if np.isnan(loss.item()):
                print("Encountered a NaN value. Skipping this outer iteration.")
                nan_encountered = True
                break  # Break out of the inner loop

            
        if nan_encountered:
            break
        loss_log.append(loss.item())
        print(f"Epoch {epoch} loss: {loss.item()/loss_scaler}")
        
    if nan_encountered:
        continue  # Continue to the next outer iteration 


    



    # set up arrays for scatterplot
    eval_latent_representations = []
    eval_reconstruction_losses = []

    # Set the model to evaluation mode
    model.eval()

    # Iterate through the dataset without computing gradients
    with torch.no_grad():
        for batch in eval_data_loader:
            x = batch[0]
            x = x.cuda()
            x_recon_pre, mu, logvar = model(x)
            ## unhot x_recon
            x_recon_pre = x_recon_pre.cpu()  
            x_recon_arr = preproc.unhot(x_recon_pre.numpy())
            x_recon = torch.from_numpy(x_recon_arr)
            # Store the latent representation and reconstruction loss
            z = model.reparameterize(mu, logvar)
            z = z.cpu()
            x = x.cpu()
            eval_latent_representations.extend(z.detach().numpy())
            eval_reconstruction_losses.extend((x - x_recon).pow(2).sum(dim=1).detach().numpy())

    eval_latent_representations = np.array(eval_latent_representations)
    eval_reconstruction_losses = np.array(eval_reconstruction_losses)

    # Normalize the reconstruction losses to the range [0, 1] for colormap
    normalized_eval_losses = (eval_reconstruction_losses / (eval_reconstruction_losses.max()))

    # create a scatter plot with the reconstruction loss
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(eval_latent_representations[:, 0], eval_latent_representations[:, 1], c=normalized_eval_losses, cmap='cubehelix', s=10)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space Visualization (Eval)')
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Reconstruction Loss')

    # create a bar graph of the reconstruction loss
    fig2, ax2 = plt.subplots()
    ax2.bar(range(len(eval_reconstruction_losses)), height=eval_reconstruction_losses)
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Reconstruction Loss (Eval)')
    ax2.set_title('Reconstruction Loss (Eval)')

    fig3, ax3 = plt.subplots()
    ax3.plot(loss_log)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss')
    ax3.set_title('Loss vs. Iteration')
    # save the plots with unique index numbers based on the current date and time
    index = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    scatter_path = f'figs/plot_{seed}_s_{index}.png'
    bar_path = f'figs/plot_{seed}_b_{index}.png'
    loss_path = f'figs/plot_{seed}_l_{index}.png'
    fig.savefig(scatter_path)
    fig2.savefig(bar_path)
    fig3.savefig(loss_path)

    latent_dx = (eval_latent_representations[:, 0].max() - eval_latent_representations[:, 0].min())
    latent_dy = (eval_latent_representations[:, 1].max() - eval_latent_representations[:, 1].min())
    latent_diameter = np.sqrt(latent_dx**2 + latent_dy**2)
    recon_loss_mean = np.mean(np.array(eval_reconstruction_losses))

    # check if a CSV file exists in the scatter folder
    csv_path = 'figs/training_info.csv'
    if os.path.isfile(csv_path):
        # load the existing CSV file into a pandas dataframe and append a new row
        traininfodf = pd.read_csv(csv_path)
        
        # should work but doesnt for some dumb reason
        new_row = {'date': index, 'batch_size': batch_size, 'kld_learning_rate': kld_learning_rate, 'mse_learning_rate': mse_learning_rate, 
                'num_epochs': num_epochs, 'seed': seed, 'latent_diameter': latent_diameter, 'recon_loss_mean': recon_loss_mean}
        traininfodf = traininfodf._append(new_row, ignore_index=True)
        print(traininfodf)
        traininfodf.to_csv(csv_path, index=False)
        #new_row = [index, batch_size, kld_learning_rate, mse_learning_rate, num_epochs, seed]
        #df.loc[len(df)+1] = new_row
    else:
        # create a new pandas dataframe with column names and add the first row
        #df = pd.DataFrame(columns=['date', 'batch_size', 'kld_learning_rate', 'mse_learning_rate', 'num_epochs', 'seed'])
        #df.loc[0] = [index, batch_size, kld_learning_rate, mse_learning_rate, num_epochs, seed]
        new_row = {'date': [index], 'batch_size': [batch_size], 'kld_learning_rate': [kld_learning_rate], 'mse_learning_rate': [mse_learning_rate], 
                'num_epochs': [num_epochs], 'seed': [seed], 'latent_diameter': [latent_diameter], 'recon_loss_mean': [recon_loss_mean]}
        traininfodf = pd.DataFrame(new_row) 
        # create the CSV file if it doesn't exist
        traininfodf.to_csv(csv_path, index=False)

    torch.save(model, f"trained_models/btcVae_{seed}.pth")

    

#plt.show()