import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MultistepSPIGOT(torch.autograd.Function):
    """
    Straight-through estimator for a categorical variable
    Forward: argmax / Backward: Softmax
    """
    @staticmethod
    def forward(ctx, s, X,   z_fixed=None):
        if z_fixed is not None:
            z = z_fixed
        else:
            z = _one_hot_argmax(s, dim=-1)
        ctx.save_for_backward(z)
        print('spigot forward')
        return z

    @staticmethod
    def backward(ctx, grad_z):
        z,  = ctx.saved_tensors
        # SPIGOT update
        print('spigot backward') 
        grad_s = z - sparsemax(z - grad_z)
        SPIGOT.forward(ctx, grad_s, -1, grad_s)

        multistep_train(X, s, z_temp, decoder, num_steps)

        return grad_s


class MultistepSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, X, y, z_fixed, decoder, num_steps):
        if z_fixed is not None:
            z = z_fixed
        else:
            z = _one_hot_argmax(s, dim=-1)
        ctx.save_for_backward(s, X, y, z, num_steps)
        # TODO: check this link: https://discuss.pytorch.org/t/when-should-you-save-for-backward-vs-storing-in-ctx/6522
        ctx.intermediate_results = decoder
        print('multistep ste forward')
        return z

    @staticmethod
    def backward(ctx, grad_z):
        s, X, y, z, num_steps,  = ctx.saved_tensors
        decoder = ctx.intermediate_results
        # gradient update
        print('multistep ste backward') 

        if num_steps > 0:
            multistep_train(X, s, z_temp, decoder, num_steps)    

        # grad_s = z - sparsemax(z - grad_z)

        return grad_s


multistep_ste = MultistepSTE.apply

def multistep_train(s, X, y, z_init, decoder, num_gradient_steps):
    # y_hat, z_hat = model(X, z)
    # s = encoder(X)
    if num_gradient_steps > 0:
        z_hat = multistep_ste(s, X, y, z_init, decoder, num_gradient_steps-1)
        if device == 'cuda':
            z_hat = z_hat.cuda()
        y_hat = decoder(X, z_hat)
        
        loss = loss_function(y_hat, y)
        loss.backward()

    return y_hat, z_hat


def train_model(model, X, y, z, X_eval, y_eval, z_eval, n_epochs, \
                save_every_steps, print_every_steps, gradient_update_steps=3):
    print(model)

    if device == 'cuda':
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.CrossEntropyLoss()
    losses = []
    eval_accuracies = []
    v_measures = []

    reward_stats = RunningStats()

    for step in range(n_epochs+1):    
        print('==================== before calling model==================')
        # y_hat, z_hat = model(X, z)
        s = model.encoder(X)
        y_hat, z_hat = multistep_train(s, X, y, z, model.decoder, )
        print('before calculating loss')
        loss = loss_function(y_hat, y)
        print('after calculating loss')

        print('before loss backward')
        loss.backward()
        print('after loss backward')

        optimizer.step()
        model.zero_grad()
        
        if step % save_every_steps == 0:
            model.set_eval()
            accuracy = eval_model(model, X_eval, y_eval, z_eval)
            model.set_train()
            accuracy = accuracy / len(y_eval)

            losses.append(loss.detach().item())
            eval_accuracies.append(accuracy)
            
            v_measure = 0
            if z_hat is not None:
                v_measure = metrics.v_measure_score(z.detach().to('cpu'), z_hat.detach().to('cpu'))
                v_measures.append(v_measure)

            if step % print_every_steps == 0:
                print('--------------------------')
                print('Step {}, loss {}, eval accuracy {}, v-measure {}'.format(step, loss, accuracy, v_measure))

    return losses, eval_accuracies, v_measures




