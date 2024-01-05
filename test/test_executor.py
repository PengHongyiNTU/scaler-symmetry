import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import unittest
from parallel import ParallelExecutor, SpecGenerator, TaskSpec

# Assume your ParallelExecutor and related classes are defined here or imported


# 1. Define a simple function for testing
def add(a, b):
    return a + b


# 2. Create a test SpecGenerator
class TestSpecGenerator(SpecGenerator):
    def __init__(self, num_tasks):
        self.num_tasks = num_tasks
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.num_tasks:
            task_id = f"task_{self.current}"
            args = [self.current, self.current + 1]
            kwargs = {}
            self.current += 1
            return TaskSpec(id=task_id, args=args, kwargs=kwargs)
        else:
            raise StopIteration


# 3. Write unit tests
class TestParallelExecutor(unittest.TestCase):
    def test_with_task_specs(self):
        executor = ParallelExecutor()
        task_specs = [
            TaskSpec(id=f"task_{i}", args=[i, i + 1], kwargs={}) for i in range(5)
        ]
        results = executor.run(func=add, task_specs=task_specs)
        self.assertIsNotNone(results, "Results should not be None")
        for result in results:
            # Extract the numeric part from the id and convert it to an integer
            task_num = int(result.id.split("_")[1])
            expected_sum = task_num + (task_num + 1)
            self.assertEqual(result.results, expected_sum)

    def test_with_spec_generator(self):
        executor = ParallelExecutor()
        spec_generator = TestSpecGenerator(5)
        results = executor.run(func=add, spec_generator=spec_generator)
        for result in results:
            task_num = int(result.id.split("_")[1])
            expected_sum = task_num + (task_num + 1)
            self.assertEqual(result.results, expected_sum)


if __name__ == "__main__":
    import ray
    unittest.main()
    ray.shutdown()
