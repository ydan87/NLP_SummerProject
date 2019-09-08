import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from core.lang import EOS_token


def show_attention(approach, input_sentence, output_words, attentions):
    """ Used to plot the attention """
    # Set up figure with color bar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       [f'<{EOS_token[1]}>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.savefig('results/' + approach + '_attention.png')


def show_loss(approach, losses):
    """ Given approach and points set, plot the points and add the approach to the filename """
    plt.figure()
    fig, ax = plt.subplots()

    ax.set_xlabel('Number of iterations (in 10k)')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss value over time ({approach})')
    plt.plot(losses)
    plt.show()
    plt.savefig(f'results/{approach}_loss_plot.png')


def show_accuracy(approach, accuracies, is_train=True):
    plt.figure()
    fig, ax = plt.subplots()

    ax.set_xlabel('Number of iterations (in 10k)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Accuracy value over time ({approach})')

    for key, values in accuracies.items():
        plt.plot(values, label=key)
    plt.legend(loc='best')
    plt.show()
    type = 'train' if is_train else 'test'
    plt.savefig(f'results/{approach}_{type}_accuracy_plot.png')
