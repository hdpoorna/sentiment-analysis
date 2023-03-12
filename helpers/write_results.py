"""
py38
hdpoorna
"""

# import packages
import os
import numpy as np
import matplotlib.pyplot as plt


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_graphs(history_dict, model_id):

    results_dir = os.path.join("results", "{}-results".format(model_id))
    make_dir(results_dir)

    epochs = range(1, len(history_dict["loss"]) + 1)

    # plot accuracy
    plt.figure()
    plt.plot(epochs, history_dict["binary_accuracy"], "b", label="Training Accuracy")
    plt.plot(epochs, history_dict["val_binary_accuracy"], "r", label="Validation Accuracy")
    plt.title('Training and Validation Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right')
    plt.xticks(epochs)
    plt.ylim(None, 1)
    fig_path = os.path.join(results_dir, "{}-accuracy.svg".format(model_id))
    plt.savefig(fig_path)
    plt.close()
    print("Accuracy graph saved to {}".format(fig_path))

    # plot loss
    plt.figure()
    plt.plot(epochs, history_dict["loss"], "b", label="Training Loss")
    plt.plot(epochs, history_dict["val_loss"], "r", label="Validation Loss")
    plt.title('Training and Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.xticks(epochs)
    plt.ylim(0, None)
    fig_path = os.path.join(results_dir, "{}-loss.svg".format(model_id))
    plt.savefig(fig_path)
    plt.close()
    print("Loss graph saved to {}".format(fig_path))


def model_summary_to_lines(model, line_length=None):
    line_lst = []
    model.summary(print_fn=lambda x: line_lst.append(x), line_length=line_length)
    return line_lst


def config_to_lines():

    config_path = os.path.join("helpers", "config.py")

    with open(config_path, "r") as f:
        config_lines = f.readlines()

    return config_lines[5:]


def history_dict_to_lines(history_dict):

    lines = []
    for k, v in history_dict.items():
        lines.append("{}: {}\n".format(k, v))

    return lines


def write_to_txt(model_id, model_summary, history_dict, test_acc, test_loss):

    summary_max_len = len(max(model_summary, key=len))

    results_dir = os.path.join("results", "{}-results".format(model_id))
    make_dir(results_dir)

    txt_path = os.path.join(results_dir, "{}.txt".format(model_id))

    with open(txt_path, "w") as f:
        # write model_id
        f.write("\nMODEL_ID: {}\n".format(model_id))
        f.write("\n{}\n".format("-" * summary_max_len))

        # write model_summary
        f.write("\nMODEL_SUMMARY\n")
        f.write("\n".join(model_summary))

        # write config
        f.write("\nCONFIG\n")
        f.write("".join(config_to_lines()))
        f.write("\n{}\n".format("-" * summary_max_len))

        # write history
        f.write("\nHISTORY\n")
        f.write("".join(history_dict_to_lines(history_dict)))
        f.write("\n{}\n".format("-" * summary_max_len))

        # write evaluations
        f.write("\nEVALUATIONS\n")
        f.write("\nTest Accuracy :{}\n".format(test_acc))
        f.write("\nTest Loss :{}\n".format(test_loss))
        f.write("\n{}\n".format("-" * summary_max_len))

    print("Text file saved to {}".format(txt_path))


if __name__ == "__main__":
    num_epochs = 10
    loss = np.random.random(num_epochs)
    val_loss = np.random.random(num_epochs)
    acc = np.random.random(num_epochs)
    val_acc = np.random.random(num_epochs)

    history = {"loss": loss,
               "val_loss": val_loss,
               "binary_accuracy": acc,
               "val_binary_accuracy": val_acc}

    save_graphs(history_dict=history, model_id="check")

    # print(config_to_lines())
