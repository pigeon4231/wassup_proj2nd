import matplotlib.pyplot as plt

def get_r2_graph(pred_prod, tst_prod, pred_cons, tst_cons, name):
    fig, axes = plt.subplots(2,2,figsize=(13,10))
    
    axes[0][0].scatter(x=pred_prod, y=tst_prod, color='b', label='r2')
    axes[0][0].set_title('dynamic prediction r2score')
    axes[0][0].set_ylabel('real')
    axes[0][0].set_xlabel('pred')

    axes[0][1].scatter(x=pred_cons, y=tst_cons, color='r', label='r2')
    axes[0][1].set_title('nomal prediction r2score')
    axes[0][1].set_ylabel('real')
    axes[0][1].set_xlabel('pred')

    axes[1][0].plot(tst_prod, color='b', label='real_y')
    axes[1][0].plot(pred_prod, color='r', label='pred_y')
    axes[1][0].set_title('dynamic prediction')
    axes[1][0].set_xlabel('time')
    axes[1][0].set_ylabel('target')
    axes[1][0].legend()

    axes[1][1].plot(tst_cons, color='b', label='real_y')
    axes[1][1].plot(pred_cons, color='r', label='pred_y')
    axes[1][1].set_title('nomal prediction')
    axes[1][1].set_xlabel('time')
    axes[1][1].set_ylabel('target')
    axes[1][1].legend()
    
    plt.legend()
    plt.show()
    plt.savefig('graph/graph_{}.png'.format(name))
    return

def get_graph(history:list,name):
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    
    axes[0].plot(history['trn_loss'], 'b-', label='trn_loss')
    axes[0].plot(history['val_loss'], 'r-', label='val_loss')
    axes[0].set_title('learning rate')
    axes[0].set_xlabel('epochs')
    axes[0].set_ylabel('loss')
    axes[1].plot(history['lr'], 'r-', label='lr')
    axes[1].set_title('nomal prediction')
    axes[1].set_xlabel('epochs')
    axes[1].set_ylabel('lr')
    plt.legend()
    plt.savefig('graph/loss_graph_{}.png'.format(name))
    return