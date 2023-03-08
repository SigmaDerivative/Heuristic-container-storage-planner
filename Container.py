class Container:
    def __init__(self, container_id: int, weight: int) -> None:
        self.container_id = container_id
        self.weight = weight

    @staticmethod
    def sort_array_weight_descending(
        containers: list["Container"],
    ) -> list["Container"]:
        return sorted(containers, key=lambda container: container.weight, reverse=True)

    @staticmethod
    def sort_array_weight_ascending(containers: list["Container"]) -> list["Container"]:
        return sorted(containers, key=lambda container: container.weight, reverse=False)

    @staticmethod
    def find_container(containers: list["Container"], con_id: int) -> "Container":
        for container in containers:
            if container.container_id == con_id:
                return container
        return None
