#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAP_SIZE 13
#define INF 1000000
#define BUFFER_SIZE 256
#define WITH_RING 1
#define WITHOUT_RING 2

typedef struct {
    int x;
    int y;
} Point;

typedef int (*CompareFunc)(const Point*, const Point*);

typedef struct {
    Point* array;
    size_t size;
    size_t capacity;
    CompareFunc compare;
    size_t heapIndex;
} Heap;
typedef struct {
    Point parent;
    int cost;
    bool visited;
    short int movemode;
    size_t heapIndex;
} Cell;
Cell map[MAP_SIZE][MAP_SIZE];
char buffer[BUFFER_SIZE];
bool isMountDetected;

size_t Left(size_t i) { return 2 * i + 1; }
size_t Right(size_t i) { return 2 * i + 2; }
size_t Parent(size_t i) { return (i - 1) / 2; }

void swap(Heap* heap, size_t i, size_t j) {
    Point temp = heap->array[i];
    heap->array[i] = heap->array[j];
    heap->array[j] = temp;

    // Обновляем heapIndex в map
    map[heap->array[i].x][heap->array[i].y].heapIndex = i;
    map[heap->array[j].x][heap->array[j].y].heapIndex = j;
}

void heapify(Heap* heap, size_t i) {
    size_t smallest = i;

    while (1) {
        size_t l = Left(i);
        size_t r = Right(i);
        smallest = i;

        if (l < heap->size &&
            heap->compare(&heap->array[l], &heap->array[smallest]) < 0) {
            smallest = l;
        }
        if (r < heap->size &&
            heap->compare(&heap->array[r], &heap->array[smallest]) < 0) {
            smallest = r;
        }

        if (smallest == i) break;

        swap(heap, i, smallest);
        i = smallest;
    }
}

void heapifyUp(Heap* heap, size_t i) {
    while (i > 0 &&
           heap->compare(&heap->array[Parent(i)], &heap->array[i]) > 0) {
        swap(heap, Parent(i), i);
        i = Parent(i);
    }
}

Heap* createHeap(CompareFunc compare) {
    Heap* heap = (Heap*)malloc(sizeof(Heap));
    if (heap == NULL) {
        fprintf(stderr, "Failed to allocate Heap struct\n");
        return NULL;
    }
    heap->size = 0;
    heap->capacity = 8;
    heap->compare = compare;
    heap->array = (Point*)malloc(heap->capacity * sizeof(Point));
    if (heap->array == NULL) {
        fprintf(stderr, "Failed to allocate heap array\n");
        free(heap);
        return NULL;
    }
    return heap;
}

void deleteHeap(Heap* heap) {
    if (heap) {
        free(heap->array);
        free(heap);
    }
}

Point extractTop(Heap* heap) {
    if (heap->size < 1) {
        return (Point){-1, -1};
    }

    Point top = heap->array[0];
    map[top.x][top.y].heapIndex = (size_t)-1;  // Убираем из кучи

    heap->array[0] = heap->array[--heap->size];
    if (heap->size > 0) {
        map[heap->array[0].x][heap->array[0].y].heapIndex = 0;
        heapify(heap, 0);
    }

    return top;
}

void insertOrUpdate(Heap* heap, Point point) {
    Cell* cell = &map[point.x][point.y];

    if (cell->heapIndex == (size_t)-1) {
        // Точки нет в куче - добавляем
        if (heap->size >= heap->capacity) {
            size_t new_capacity = heap->capacity * 2;
            Point* new_array =
                (Point*)realloc(heap->array, new_capacity * sizeof(Point));
            if (new_array == NULL) return;
            heap->array = new_array;
            heap->capacity = new_capacity;
        }

        heap->array[heap->size] = point;
        cell->heapIndex = heap->size;
        heapifyUp(heap, heap->size);
        heap->size++;
    } else {
        // Точка уже в куче - обновляем позицию
        heapifyUp(heap, cell->heapIndex);
        heapify(heap, cell->heapIndex);
    }
}

int r = 0;
Point gollum = {0};

int min(int a, int b) { return a > b ? b : a; }
int max(int a, int b) { return a > b ? a : b; }
bool isObstacle(char type) {
    switch (type) {
        case 'R':
        case 'C':
        case 'M':
        case 'G':
            return false;
        default:
            return true;
    }
}

int compare(const Point* a, const Point* b) {
    if (a->x == b->x && a->y == b->y) {
        return 1;
    }
    int manhattan_a = abs(a->x - gollum.x) + abs(a->y - gollum.y);
    int cost_a = map[a->x][a->y].cost + manhattan_a;

    int manhattan_b = abs(b->x - gollum.x) + abs(b->y - gollum.y);
    int cost_b = map[b->x][b->y].cost + manhattan_b;

    if (cost_a != cost_b) {
        return cost_a - cost_b;
    }

    return manhattan_a - manhattan_b;
}

void initMap() {
    for (int i = 0; i < MAP_SIZE; i++) {
        for (int j = 0; j < MAP_SIZE; j++) {
            map[i][j] = (Cell){{-1, -1}, INF, false, 0, (size_t)-1};
        }
    }
}
// Функция для обновления одного соседа
void updateNeighbor(Heap* heap, Point parent, int dx, int dy, bool isObstacle,
                    short int moveMode) {
    int new_x = parent.x + dx;
    int new_y = parent.y + dy;

    // Проверка границ
    if (!isObstacle || new_x < 0 || new_x >= MAP_SIZE || new_y < 0 ||
        new_y >= MAP_SIZE) {
        return;
    }

    // Проверка и обновление стоимости
    int new_cost = map[parent.x][parent.y].cost + 1;
    if (new_cost < map[new_x][new_y].cost) {
        map[new_x][new_y].cost = new_cost;
        map[new_x][new_y].parent = parent;
        map[new_x][new_y].movemode = moveMode;
        insertOrUpdate(heap, (Point){new_x, new_y});
    } else if (new_cost == map[new_x][new_y].cost &&
               parent.x == map[new_x][new_y].parent.x &&
               parent.y == map[new_x][new_y].parent.y) {
        map[new_x][new_y].movemode |= moveMode;
    }
}

// Функция для чтения и обработки препятствий
bool readObstacles(Point parent, bool moves[]) {
    // Читаем строку целиком
    if (fgets(buffer, BUFFER_SIZE, stdin) == NULL) {
        return false;
    }

    // Убираем символ новой строки
    buffer[strcspn(buffer, "\n")] = '\0';

    // Проверяем специальную команду
    if (!isMountDetected && strstr(buffer, "My precious!") != NULL) {
        // Парсим новые координаты Голлума
        sscanf(buffer, "My precious! Mount Doom is %d %d", &gollum.x,
               &gollum.y);
        return true;  // Возвращаем true чтобы показать что была специальная
                      // команда
    }

    // Если не специальная команда, парсим количество препятствий
    int p;
    sscanf(buffer, "%d", &p);

    // Инициализация всех направлений как разрешенных
    for (int i = 0; i < 4; i++) {
        moves[i] = true;
    }

    // Чтение препятствий
    for (int i = 0; i < p; i++) {
        int x, y;
        char type;
        fgets(buffer, BUFFER_SIZE, stdin);
        sscanf(buffer, "%d %d %c", &x, &y, &type);

        // Определение направления и установка флага
        if (x == parent.x + 1 && y == parent.y) {
            moves[0] = !isObstacle(type);
        } else if (x == parent.x && y == parent.y + 1) {
            moves[1] = !isObstacle(type);
        } else if (x == parent.x - 1 && y == parent.y) {
            moves[2] = !isObstacle(type);
        } else if (x == parent.x && y == parent.y - 1) {
            moves[3] = !isObstacle(type);
        }
    }

    return false;  // Обычные препятствия, не специальная команда
}

void goToStart(Point parent) {
    if (parent.x == 0 && parent.y == 0) {
        return;
    }
    while (1) {
        parent = map[parent.x][parent.y].parent;
        printf("%d %d\n", parent.x, parent.y);
        fflush(stdout);
        if (parent.x == 0 && parent.y == 0) {
            break;
        }
        readObstacles(parent, (bool[4]){});
    }
}
short int switchMoveMode(short int moveMode) {
    return moveMode == WITH_RING ? WITHOUT_RING : WITH_RING;
}

bool checkAndSwitchMode(Point parent, short int* currentMoveMode) {
    bool haveCorrectMode =
        (*currentMoveMode & map[parent.x][parent.y].movemode);

    if (!haveCorrectMode) {
        *currentMoveMode = switchMoveMode(*currentMoveMode);

        if (*currentMoveMode == WITH_RING) {
            printf("r\n");
        } else {
            printf("rr\n");
        }
        fflush(stdout);
        // Читаем препятствия после переключения
        bool dummyMoves[4];
        readObstacles(parent, dummyMoves);

        return true;  // Режим был переключен
    }

    return false;  // Режим не менялся
}

void goToPoint(Point parent, Point dest) {
    // Сначала идем к старту (0,0)
    goToStart(parent);
    readObstacles(parent, (bool[4]){});

    // Теперь мы в (0,0), строим путь до dest
    Point path[MAP_SIZE * MAP_SIZE];
    int path_length = 0;

    // Собираем путь от dest до (0,0) через parent
    Point current = dest;
    while (!(current.x == 0 && current.y == 0)) {
        path[path_length++] = current;
        current = map[current.x][current.y].parent;

        if (path_length >= MAP_SIZE * MAP_SIZE) {
            fprintf(stderr, "Path too long\n");
            return;
        }
    }

    short int currentMoveMode = WITHOUT_RING;

    // Выводим путь в обратном порядке (от старта к dest)
    for (int i = path_length - 1; i >= 0; i--) {
        checkAndSwitchMode(path[i], &currentMoveMode);
        printf("%d %d\n", path[i].x, path[i].y);
        fflush(stdout);
        // Читаем препятствия для всех точек кроме последней
        if (i > 0) {
            readObstacles(path[i], (bool[4]){});
        }
    }
}

// Основная функция обновления соседей
bool updateNeighbours(Heap* heap, Point parent, short int moveMode) {
    bool moves[4];  // down, right, up, left

    // Читаем и обрабатываем препятствия
    bool isSpecialCommand = readObstacles(parent, moves);

    if (!isSpecialCommand) {
        // Обновляем всех соседей
        updateNeighbor(heap, parent, 1, 0, moves[0], moveMode);   // down
        updateNeighbor(heap, parent, 0, 1, moves[1], moveMode);   // right
        updateNeighbor(heap, parent, -1, 0, moves[2], moveMode);  // up
        updateNeighbor(heap, parent, 0, -1, moves[3], moveMode);  // left
    }

    return isSpecialCommand;
}

void reset(Heap* heap, Point* parent, short int* currentMoveMode) {
    goToStart(*parent);
    initMap();
    deleteHeap(heap);
    heap = createHeap(compare);
    if (heap) {
        map[0][0].cost = 0;
        map[0][0].movemode = WITH_RING | WITHOUT_RING;
        map[0][0].visited = true;
    }
    *parent = (Point){0, 0};
    *currentMoveMode = WITHOUT_RING;
    isMountDetected = true;
}

int main() {
    // Поменять логику с moveMode, чтобы мы не давали новые права а скорее
    // ограничивали изначальные

    Heap* heap = createHeap(compare);
    if (!heap) {
        fprintf(stderr, "Failed to create heap\n");
        return 1;
    }
    initMap();
    map[0][0].cost = 0;
    map[0][0].visited = true;
    map[0][0].movemode = WITH_RING | WITHOUT_RING;

    fgets(buffer, BUFFER_SIZE, stdin);
    sscanf(buffer, "%d", &r);
    fgets(buffer, BUFFER_SIZE, stdin);
    sscanf(buffer, "%d %d", &gollum.x, &gollum.y);

    Point parent = {0, 0};
    short int currentMoveMode = WITHOUT_RING;
    isMountDetected = false;
    do {
        if (isMountDetected && parent.x == gollum.x && parent.y == gollum.y) {
            printf("e %d", map[parent.x][parent.y].cost);
            break;
        }
        bool wasSpecialCommand =
            updateNeighbours(heap, parent, currentMoveMode);
        if (wasSpecialCommand) {
            reset(heap, &parent, &currentMoveMode);
            updateNeighbours(heap, parent, currentMoveMode);
        }

        if (switchMoveMode(currentMoveMode) &
            map[parent.x][parent.y].movemode) {
            currentMoveMode = switchMoveMode(currentMoveMode);
            if (currentMoveMode == WITH_RING) {
                printf("r\n");
            } else {
                printf("rr\n");
            }
            fflush(stdout);
            wasSpecialCommand = updateNeighbours(heap, parent, currentMoveMode);
            if (wasSpecialCommand) {
                reset(heap, &parent, &currentMoveMode);
                updateNeighbours(heap, parent, currentMoveMode);
            }
        }
        map[parent.x][parent.y].visited = true;
        Point next = parent;

        while (map[next.x][next.y].visited && heap->size > 0) {
            next = extractTop(heap);
        }
        if (!map[next.x][next.y].visited) {
            Point parentOfNext = map[next.x][next.y].parent;
            if (parent.x == parentOfNext.x && parent.y == parentOfNext.y) {
                checkAndSwitchMode(next, &currentMoveMode);
                printf("%d %d\n", next.x, next.y);
                fflush(stdout);
            } else {
                goToPoint(parent, next);
            }
            parent = next;
        }
    } while (heap->size > 0);
    deleteHeap(heap);
    if (map[gollum.x][gollum.y].cost >= INF || !isMountDetected) printf("e -1\n");
    fflush(stdout);
    return 0;
}
