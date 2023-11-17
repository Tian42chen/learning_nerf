class Renderer():
    def __init__(self, net) -> None:
        self.net = net
        
    def render(self, batch):
        return self.net(batch)