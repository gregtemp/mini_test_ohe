import torch
from pythonosc import osc_server
from pythonosc import dispatcher
from pythonosc import udp_client
import argparse
import pandas as pd
import itertools

# Get the column names
df = pd.read_csv("operator-presets2-norm_mod.csv")
column_names = df.columns.str.strip().tolist()

model = torch.load("trained_models/betaVae_42")
model.eval()

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="127.0.0.1",
    help="The ip of the OSC server")
parser.add_argument("--port", type=int, default=9001,
    help="The port the OSC server is listening on")
args = parser.parse_args()
# Set up OSC client (for sending messages)
client = udp_client.SimpleUDPClient(args.ip, args.port)
client.send_message("/status", 1)

def handle_message(unused_addr, *values):
    # Convert the received values to a PyTorch tensor
    input_data = torch.tensor(values, dtype=torch.float32).unsqueeze(0)  # unsqueeze(0) to add batch dimension
    print(f"/input - {values}")
    # Send the input data to the model and get the output
    with torch.no_grad():  # Don't track gradients during prediction
        output = model.decoder(input_data)

    # Convert the output to a list and send it back over OSC
    output_list = output.squeeze(0).tolist()  # squeeze(0) to remove batch dimension
    interlaced = [[name, value] for name, value in zip(column_names, output_list)]
    flattened = list(itertools.chain(*interlaced))
    client.send_message("/prediction", flattened)

# Set up OSC server (for receiving messages)
dispatcher = dispatcher.Dispatcher()
dispatcher.map("/input", handle_message)


try:
    server = osc_server.ThreadingOSCUDPServer(('localhost', 9000), dispatcher)
    print("Serving on {}".format(server.server_address))
    client.send_message("/status", 2)
    server.serve_forever()
except KeyboardInterrupt:
    print("\nShutting down OSC Server...")
    server.shutdown()
    server.server_close()
    client.send_message("/status", 0)
    print("OSC Server shut down successfully.")

