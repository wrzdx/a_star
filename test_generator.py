# correct_test_generator.py
import json
import random
import os
from typing import List, Tuple, Dict, Any

MAP_SIZE: int = 13

def neuman_zone(x: int, y: int, radius: int) -> List[Tuple[int, int]]:
    """Зона Неймана (крест)"""
    cells = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if abs(i) + abs(j) <= radius:
                cells.append((x + i, y + j))
    return cells

def moore_zone(x: int, y: int, radius: int, with_ears: bool) -> List[Tuple[int, int]]:
    """Зона Мура (квадрат) с ушами или без"""
    cells = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            cells.append((x + i, y + j))
    
    if with_ears:
        # Добавляем "уши" - угловые клетки за пределами основного квадрата
        cells.extend([
            (x - radius - 1, y - radius - 1),
            (x - radius - 1, y + radius + 1),
            (x + radius + 1, y - radius - 1),
            (x + radius + 1, y + radius + 1),
        ])
    return cells

def get_zone(x: int, y: int, token: str, with_ring: bool) -> List[Tuple[int, int]]:
    """Возвращает зону влияния токена"""
    if token == "O":  # Орк
        # Орк: радиус 1 без кольца, радиус 0 с кольцом
        radius = 0 if with_ring else 1
        return neuman_zone(x, y, radius)
    
    elif token == "U":  # Урук-хай
        # Урук-хай: радиус 2 без кольца, радиус 1 с кольцом  
        radius = 1 if with_ring else 2
        return neuman_zone(x, y, radius)
    
    elif token == "N":  # Назгул
        if with_ring:
            # С кольцом: квадрат 5x5 (радиус 2)
            return moore_zone(x, y, 2, False)
        else:
            # Без кольца: квадрат 3x3 (радиус 1) + уши
            return moore_zone(x, y, 1, True)
    
    elif token == "W":  # Сторожевая башня
        # Всегда квадрат 5x5 (радиус 2), уши зависят от кольца
        return moore_zone(x, y, 2, with_ring)
    
    return [(x, y)]

def get_all_danger_cells(tokens: List[Dict]) -> set[Tuple[int, int]]:
    """Возвращает все опасные клетки от всех врагов"""
    danger_cells = set()
    for token in tokens:
        if token["type"] in ["O", "U", "N", "W"]:
            # Добавляем зоны с кольцом и без кольца
            zone_with_ring = get_zone(token["x"], token["y"], token["type"], True)
            zone_without_ring = get_zone(token["x"], token["y"], token["type"], False)
            
            # Фильтруем клетки за пределами карты
            for cell in zone_with_ring + zone_without_ring:
                if 0 <= cell[0] < MAP_SIZE and 0 <= cell[1] < MAP_SIZE:
                    danger_cells.add(cell)
    return danger_cells

def is_position_safe(x: int, y: int, danger_cells: set) -> bool:
    """Проверяет, безопасна ли позиция"""
    return (x, y) not in danger_cells

def generate_random_position(excluded_positions: set) -> Tuple[int, int]:
    """Генерирует случайную позицию, исключая занятые"""
    while True:
        x, y = random.randint(0, MAP_SIZE - 1), random.randint(0, MAP_SIZE - 1)
        if (x, y) not in excluded_positions:
            return (x, y)

def generate_test_case(seed: int = None) -> Dict[str, Any]:
    """Генерирует один тестовый случай с правильной проверкой безопасности"""
    if seed is not None:
        random.seed(seed)
    
    max_attempts = 50
    for attempt in range(max_attempts):
        occupied_positions = set()
        tokens = []
        
        # Стартовая позиция всегда (0, 0)
        start_pos = (0, 0)
        occupied_positions.add(start_pos)
        
        # 1. Сначала размещаем врагов
        enemy_types = ["W", "U"] + ["N"] * random.randint(0, 1) + ["O"] * random.randint(1, 2)
        enemies = []
        
        for enemy_type in enemy_types:
            enemy_placed = False
            for enemy_attempt in range(20):
                pos = generate_random_position(occupied_positions)
                
                # Временно добавляем врага
                temp_enemies = enemies + [{"type": enemy_type, "x": pos[0], "y": pos[1]}]
                temp_danger_cells = get_all_danger_cells(temp_enemies)
                
                # Проверяем что стартовая позиция безопасна
                if is_position_safe(0, 0, temp_danger_cells):
                    enemy_data = {"type": enemy_type, "x": pos[0], "y": pos[1]}
                    enemies.append(enemy_data)
                    tokens.append(enemy_data)
                    occupied_positions.add(pos)
                    enemy_placed = True
                    break
            
            if not enemy_placed:
                break  # Не удалось разместить врага, начинаем заново
        
        # Если не удалось разместить всех врагов, пробуем снова
        if len(enemies) < len(enemy_types):
            continue
        
        # Получаем финальные опасные клетки после размещения всех врагов
        final_danger_cells = get_all_danger_cells(enemies)
        
        # 2. Теперь размещаем G, M, C в безопасных позициях
        required_tokens = ["G", "M", "C"]
        important_tokens_placed = True
        
        for token_type in required_tokens:
            token_placed = False
            for token_attempt in range(30):
                pos = generate_random_position(occupied_positions)
                
                # Проверяем что позиция безопасна
                if is_position_safe(pos[0], pos[1], final_danger_cells):
                    token_data = {"type": token_type, "x": pos[0], "y": pos[1]}
                    tokens.append(token_data)
                    occupied_positions.add(pos)
                    token_placed = True
                    break
            
            if not token_placed:
                important_tokens_placed = False
                break
        
        if not important_tokens_placed:
            continue  # Не удалось разместить G, M, C, пробуем снова
        
        # 3. Случайный радиус восприятия
        radius = random.randint(1, 2)
        
        # Финальная проверка безопасности
        if validate_test_case({"tokens": tokens}):
            return {
                "radius": radius,
                "tokens": tokens,
                "map_size": MAP_SIZE
            }
    
    # Если не удалось сгенерировать валидный тест после всех попыток
    return None

def validate_test_case(test_case: Dict) -> bool:
    """Проверяет валидность тестового случая"""
    try:
        # Проверяем обязательные токены
        required_types = {"G", "M", "C"}
        found_types = {token["type"] for token in test_case["tokens"]}
        
        if not required_types.issubset(found_types):
            return False
        
        # Получаем опасные клетки
        danger_cells = get_all_danger_cells(test_case["tokens"])
        
        # Проверяем безопасность стартовой позиции
        if not is_position_safe(0, 0, danger_cells):
            return False
        
        # Проверяем безопасность G, M, C
        for token in test_case["tokens"]:
            if token["type"] in ["G", "M", "C"]:
                if not is_position_safe(token["x"], token["y"], danger_cells):
                    return False
        
        # Проверяем что все позиции в пределах карты
        for token in test_case["tokens"]:
            x, y = token["x"], token["y"]
            if not (0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE):
                return False
        
        return True
        
    except Exception:
        return False

def generate_test_suite(num_tests: int, output_file: str = "generated_tests.json", seed: int = None):
    """Генерирует набор тестов и сохраняет в JSON файл"""
    if seed is not None:
        random.seed(seed)
    
    test_suite = {
        "metadata": {
            "total_tests": num_tests,
            "map_size": MAP_SIZE,
            "generated_with_seed": seed,
            "description": "Correct test cases with proper safety validation"
        },
        "tests": []
    }
    
    valid_tests = 0
    total_attempts = 0
    
    while valid_tests < num_tests and total_attempts < num_tests * 10:
        total_attempts += 1
        test_case = generate_test_case()
        
        if test_case is not None and validate_test_case(test_case):
            test_case["test_id"] = valid_tests
            test_suite["tests"].append(test_case)
            valid_tests += 1
            
            if valid_tests % 100 == 0:
                print(f"Generated {valid_tests}/{num_tests} valid test cases...")
    
    if valid_tests < num_tests:
        print(f"Warning: Only generated {valid_tests}/{num_tests} valid test cases after {total_attempts} attempts")
    else:
        print(f"Successfully generated {valid_tests} valid test cases")
    
    # Сохраняем в файл
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_suite, f, indent=2, ensure_ascii=False)
    
    # Выводим статистику
    token_counts = {}
    for test in test_suite["tests"]:
        for token in test["tokens"]:
            token_type = token["type"]
            token_counts[token_type] = token_counts.get(token_type, 0) + 1
    
    print("\nToken statistics:")
    for token_type, count in sorted(token_counts.items()):
        avg_per_test = count / valid_tests
        print(f"  {token_type}: {avg_per_test:.2f} per test")

def debug_test_case(test_case: Dict):
    """Отладочная информация о тестовом случае"""
    print(f"Test case {test_case.get('test_id', 'unknown')}:")
    
    danger_cells = get_all_danger_cells(test_case["tokens"])
    
    for token in test_case["tokens"]:
        status = "SAFE" if is_position_safe(token["x"], token["y"], danger_cells) else "DANGER"
        print(f"  {token['type']} at ({token['x']}, {token['y']}): {status}")
    
    print(f"  Start (0,0): {'SAFE' if is_position_safe(0, 0, danger_cells) else 'DANGER'}")
    print(f"  Total danger cells: {len(danger_cells)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate correct test cases for Frodo pathfinding")
    parser.add_argument("--num-tests", type=int, default=1000, help="Number of test cases to generate")
    parser.add_argument("--output", type=str, default="generated_tests.json", help="Output JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true", help="Debug first test case")
    
    args = parser.parse_args()
    
    generate_test_suite(args.num_tests, args.output, args.seed)
    
    if args.debug:
        with open(args.output, 'r') as f:
            test_suite = json.load(f)
        if test_suite["tests"]:
            debug_test_case(test_suite["tests"][0])