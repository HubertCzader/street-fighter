class Driver(Protocol):
    @abstractmethod
    def start_attempt(self, state: State) -> Action:
        raise NotImplementedError

    @abstractmethod
    def start_optimal(self, state: State) -> Action:
        raise NotImplementedError

    @abstractmethod
    def control(self, state: State, last_reward: int) -> Action:
        raise NotImplementedError

    @abstractmethod
    def finished_learning(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def control_optimal(self, state: State, last_reward: int) -> Action:
        raise NotImplementedError

