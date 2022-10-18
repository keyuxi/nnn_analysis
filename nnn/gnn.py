"""
Define the training pipeline
"""
from torch.nn import Linear, BatchNorm1d, ReLU, Sequential, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATv2Conv, BatchNorm, TransformerConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn.aggr import Set2Set

def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(project="NNN_GNN", config=hyperparameters):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, train_loader, test_loader, criterion, optimizer = make(config)
      print(model)

      # and use them to train the model
      train(model, train_loader, test_loader, criterion, optimizer, config)

      # and test its final performance
      test(model, train_loader, test_loader)

    return model


def make(config):
    # Make the data
    torch.manual_seed(12345)
    if config['dataset'] == 'NNN_v0':
        dataset = NNNDataset(root='/content/drive/My Drive/Colab Notebooks/data/nnn')
    has_gpu = torch.cuda.is_available()
    train_loader = DataLoader(dataset.train_set, batch_size=config['batch_size'],
                            shuffle=True, pin_memory=has_gpu)
    test_loader = DataLoader(getattr(dataset, config['mode']+'_set'), batch_size=config['batch_size'],
                            shuffle=False, pin_memory=has_gpu)

    # Make the model
    if config['architecture'] == 'GraphTransformer':
        model = GTransformer(config).to(device)

    # Make the loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config'learning_rate'])
    
    return model, train_loader, test_loader, criterion, optimizer


class GTransformer(torch.nn.Module):
    def __init__(self, config):
        super(GTransformer, self).__init__()
        torch.manual_seed(12345)
        num_node_features = 4
        num_edge_features = 3
        num_params = 2
        num_heads = 1
        self.graphconv_dropout = config['graphconv_dropout']
        self.linear_dropout = config['linear_dropout']

        self.convs = ModuleList([
            TransformerConv(num_node_features, config['hidden_channels'],
                               heads=num_heads, edge_dim=num_edge_features)] +
            [TransformerConv(config['hidden_channels'], config['hidden_channels'],
                               heads=num_heads, edge_dim=num_edge_features)] *
            (config['n_graphconv_layer'] - 1))

        self.norm = BatchNorm(in_channels=config['hidden_channels'])

        linear_list = []
        if config['pooling'] == 'Set2Set':
            self.aggr = Set2Set(config['hidden_channels'], 
                                processing_steps=config['processing_steps'])
            linear_list.append(Linear(2*config['hidden_channels'], config['linear_hidden_channels'][0]))
        elif config['pooling'] == 'global_add_pool':
            self.aggr = global_add_pool
            linear_list.append(Linear(config['hidden_channels'], config['linear_hidden_channels'][0]))
        else:
            raise 'Invalid config.pooling %s' % config['pooling']
        
        if config['n_linear_layer'] > 2:
            linear_list.extend([Linear(config['linear_hidden_channels'][i - 1], 
                                    config['linear_hidden_channels'][i])
                                for i in range(1, config['n_linear_layer'])])
        linear_list.append(Linear(config['linear_hidden_channels'][-1], 
                                  num_params))
        self.linears = ModuleList(linear_list)

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Obtain node embeddings
        for i, l in enumerate(self.convs):
            x = l(x, edge_index, edge_attr)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.graphconv_dropout, training=self.training)

        # 2. Pooling layer
        x = self.aggr(x, batch)

        # 3. Apply a final regressor
        for i, l in enumerate(self.linears):
            x = l(x)
            x = F.relu(x)
            if i < len(self.linears) - 1:
                x = F.dropout(x, p=self.linear_dropout, training=self.training)

        x = torch.flatten(x)
        
        return x


def train_epoch(model, train_loader, criterion, optimizer, config):
    """
    Train one epoch, called by train()
    """

    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.

        out = model(data.x.to(device), data.edge_index.to(device), 
                    data.edge_attr.to(device), data.batch.to(device))  # Perform a single forward pass.

        loss = criterion(out, data.y.to(device))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def get_loss(loader, model):
    """
    Gets the loss at log points during training
    """
    model.eval()

    rmse = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x.to(device), data.edge_index.to(device), 
                    data.edge_attr.to(device), data.batch.to(device)) 
        rmse += float(((out - data.y.to(device))**2).sum())  # Check against ground-truth labels.
    return np.sqrt(rmse / len(loader.dataset))  # Derive ratio of correct predictions.


def unorm(arr):
    arr[:,0] = unorm_dH(arr[:,0]) 
    arr[:,1] = unorm_Tm(arr[:,1])
    return arr


def model_pred(model, data, device='cuda:0'):
    """
    handy function to predict one sequence and convert the result to (1,2) np.array
    """
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device), 
                data.edge_attr.to(device), data.batch.to(device))
    return out.to('cpu').detach().numpy().reshape(-1, 2)


def get_truth_pred(loader, model):

    y = np.zeros((0,2))
    pred = np.zeros((0,2))
    model.eval()
    for i, data in enumerate(loader):
        y = np.concatenate((y, data.y.detach().numpy().reshape(-1,2)), axis=0)
        out = model_pred(model, data)
        pred = np.concatenate((pred, out), axis=0)

    y = unorm(y)
    pred = unorm(pred)

    return dict(y=y, pred=pred)

def plot_truth_pred(result, ax, param='dH', title='Train'):
    color_dict = dict(dH='c', Tm='cornflowerblue', dG_37='teal', dS='steelblue')
    if param == 'dH':
        col = 0
        lim = [-55, -5]
    elif param == 'Tm':
        col = 1
        lim = [20, 60]

    c = color_dict[param]

    y, pred = result['y'][:, col], result['pred'][:, col]
    ax.scatter(y, pred, c=c, marker='D', alpha=.05)
    rmse = np.sqrt(np.mean((y - pred)**2))
    mae = np.mean(np.abs(y - pred))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('measured ' + param)
    ax.set_ylabel('predicted ' + param)
    ax.set_title('%s: RMSE = %.3f, MAE = %.3f' % (title, rmse, mae))

    return rmse, mae


def train(model, train_loader, test_loader, criterion, optimizer, config):
    """
    Calls train_epoch() and logs
    """
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    every_n_epoch = 10
    for epoch in range(config['n_epoch']):
        train_epoch(model, train_loader, criterion, optimizer, config)
        if epoch % every_n_epoch == 0:
            train_rmse = get_loss(train_loader, model)
            test_rmse = get_loss(test_loader, model)
            wandb.log({"train_rmse": train_rmse,
                    "test_rmse": test_rmse})
            print(f'Epoch: {epoch:03d}, Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}')


def test(model, train_loader, test_loader):
    train_result = get_truth_pred(train_loader, model)
    test_result = get_truth_pred(test_loader, model)

    fig, ax = plt.subplots(2, 2, figsize=(12,12))
    _ = plot_truth_pred(train_result, ax[0,0], param='dH')
    _ = plot_truth_pred(train_result, ax[0,1], param='Tm')
    dH_rmse, dH_mae = plot_truth_pred(test_result, ax[1,0], param='dH', title='Validation')
    Tm_rmse, Tm_mae = plot_truth_pred(test_result, ax[1,1], param='Tm', title='Validation')

    wandb.run.summary["dH_rmse"] = dH_rmse
    wandb.run.summary["dH_mae"] = dH_mae
    wandb.run.summary["Tm_rmse"] = Tm_rmse
    wandb.run.summary["Tm_mae"] = Tm_mae

    plt.show()