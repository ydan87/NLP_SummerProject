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


def show_loss(approach, iterations, losses):
    """ Given approach and points set, plot the points and add the approach to the filename """
    fig = plt.figure()
    plt.title(f'{approach.upper()} - Loss value over time')
    plt.xlabel = 'Number of iterations'
    plt.ylabel = 'Loss'
    plt.plot(iterations, losses)

    plt.show()
    fig.savefig(f'results/{approach}_loss_plot.png')


def show_accuracy(approach, iterations, accuracies, is_train=True):
    fig = plt.figure()

    plt.xlabel = f'Number of iterations'
    plt.ylabel = 'Accuracy (%)'
    plt.title(f'{approach.upper()} - Accuracy value over time')

    for key, values in accuracies.items():
        plt.plot(iterations, values, label=key)
    plt.legend(loc='best')
    plt.show()
    type = 'train' if is_train else 'test'
    fig.savefig(f'results/{approach}_{type}_accuracy_plot.png')
