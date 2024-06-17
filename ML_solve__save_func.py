import numpy as np
import matplotlib.pyplot as plt
import os

def save_plot(i, image_folder, yhat_np_list, y_np, x_fine_np, loss_array, min_value, max_value):

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # solution comparison plot
    axs[0].plot(x_fine_np, y_np, '--', linewidth=3, label='Actual')
    axs[0].plot(x_fine_np, yhat_np_list[i],label='PINN')
    axs[0].set_title('Actual Solution vs PINN Solution')
    axs[0].set_xlabel('$x$')
    axs[0].set_ylabel('$y$')
    axs[0].set_ylim(min_value, max_value)
    axs[0].legend()

    # loss function plot
    axs[1].axvline(i, color='red', linestyle='-', linewidth=1)
    axs[1].scatter(i, loss_array[i], color='red', s=15) 
    axs[1].semilogy(loss_array, color='blue')  
    axs[1].set_ylabel('$Loss$')
    axs[1].set_xlabel('$Epoch$')
    axs[1].set_title('Loss Function Evaluation at Each Epoch')

    file_name = f'frame_{i:04d}.png'  
    plt.savefig(os.path.join(image_folder, file_name))
    plt.close()