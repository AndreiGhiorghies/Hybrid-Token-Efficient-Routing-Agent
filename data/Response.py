class Response:
    def __init__(self, id, success = False, answer = "", route = None, model = None):
        self.id = id
        self.success = success
        self.answer = answer
        self.model = model
        self.route = route
        