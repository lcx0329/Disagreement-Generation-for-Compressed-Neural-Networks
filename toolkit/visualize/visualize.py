import matplotlib as mpl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# def imshow(image_group, titles=None):
#     b = 2
#     fig, axes = plt.subplots(len(image_group), len(image_group[0]), figsize=(len(image_group[0])*b, len(image_group)*b))
#     for i, images in enumerate(image_group):
#         for idx, image in enumerate(images):
#             axes[i][idx].imshow(image)
#             axes[i][idx].set_axis_off()
            
#         if titles:
#             axes[i][idx].set_suptitle(titles[i])

#     fig.show()

def imshow(image_group, titles=None):
    b = 2
    fig, axes = plt.subplots(len(image_group), len(image_group[0]), figsize=(len(image_group[0])*b, len(image_group)*b))
    for i, images in enumerate(image_group):
        for idx, image in enumerate(images):
            ax = axes[i][idx]
            ax.imshow(image)
            ax.set_axis_off()

            # Add titles to the first subplot in each row
            if idx == 0 and titles is not None:
                ax.set_title(titles[i], rotation='vertical', ha='right')

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()


def heatmap():
    plt.imshow(cmap="inferno")
    
    

label = ["class"+str(i) for i in range(10)]

def create_multi_bars(labels, datas, title, tick_step=1, group_gap=0.2, bar_gap=0):

    
    plt.figure(figsize=(10, 5))
    x = np.arange(len(labels)) * tick_step
    
    group_num = len(datas)
    
    group_width = tick_step - group_gap
    
    bar_span = group_width / group_num
    
    bar_width = bar_span - bar_gap
    
    for index, y in enumerate(datas):
        plt.bar(x + index*bar_span, y, bar_width)
    plt.ylabel('Acc Difference')
    plt.title(title)
    
    ticks = x + (group_width - bar_span) / 2
    plt.xticks(ticks, labels)
    # plt.ylim(80, 100)
    # plt.show()
    plt.savefig("{}.png".format(title), bbox_inches='tight', transparent=False)
