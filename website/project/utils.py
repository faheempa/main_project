from defines import ESN, LSTM, Agent
import torch
import warnings
warnings.filterwarnings("ignore")

def get_agent():
    esn_input_size = 4
    esn_reservoir_size = 1024
    esn_output_size = 128
    lstm_input_size = 128
    lstm_num_hidden = 512
    lstm_num_layers = 4
    lstm_output_size = 3

    esn = ESN(esn_input_size, esn_reservoir_size, esn_output_size)
    esn.load_state_dict(torch.load('esn.pt'))
    lstm = LSTM(lstm_input_size, lstm_num_hidden, lstm_num_layers, lstm_output_size, esn)
    lstm.load_state_dict(torch.load('lstm.pt'))
    crypto_agent = Agent(lstm, "test_data.csv")
    state = crypto_agent.env.reset()
    return crypto_agent, state

def get_action(agent, state, amount):
    try:
        profit = torch.tensor(0)
        action = agent.get_action(state)
        new_state, reward, done, change = agent.env.step(action)
        profit = change/agent.env.price*amount
        agent.remember(state, action, reward)
        agent.train_short_memory(state, action, reward)
        state = new_state
        if done:
            agent.train_long_memory()
            print("Training long memory")
    except StopIteration:
        pass
    except Exception as e:
        pass
    finally:
        return action, state, profit