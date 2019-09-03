import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from core.lang import EOS_token


def show_attention(approach, input_sentence, output_words, attentions):
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


def show_plot(approach, points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.xlabel = 'Number of iterations (in 10k)'
    plt.ylabel = 'Loss (%)'
    plt.title = 'Loss value over time (' + approach + ')'
    plt.plot(points)
    plt.savefig('results/' + approach + '_loss_plot.png')
