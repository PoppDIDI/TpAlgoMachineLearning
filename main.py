import pygame, sys, copy, time, random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------
# Paramètres globaux du jeu
# ---------------------------
WIDTH, HEIGHT = 600, 700           # Plateau : 600x600, barre d'état en bas (100px)
ROWS, COLS = 3, 3                  # Grille 3x3
CELL_SIZE = WIDTH // COLS

# 🎨 Nouvelle palette de couleurs moderne
BG_COLOR = (30, 30, 40)            # Fond sombre (ancien bois remplacé)
GRID_COLOR = (100, 100, 120)        # Gris pour la grille
BLUE_PIECE = (200, 220, 255)          # Pièces noires modernisées
RED_PIECE = (200, 0, 0)       # Pièces blanches modernisées
SELECTED_COLOR = (255, 180, 0)      # Jaune-orangé vif pour la sélection
STATUS_BG = (50, 50, 70)            # Fond de la barre de statut
STATUS_TEXT = (255, 255, 255)       # Texte blanc pour lisibilité

# 🎨 Design popup
POPUP_BG_COLOR = (20, 20, 30, 200)  # Fond semi-transparent
BUTTON_COLOR = (100, 200, 250)      # Bleu clair pour le bouton


# Initialisation de la police
pygame.font.init()
FONT = pygame.font.SysFont("arial", 28)
POPUP_FONT = pygame.font.SysFont("arial", 36)

# ---------------------------
# Fonctions globales de plateau
# ---------------------------
def board_coords(i, j):
    """
    Conversion des indices de la grille (0 à 2) en coordonnées centrées
    où la case centrale (1,1) correspond à (0, 0)
    """
    return j - 1, i - 1

def is_adjacent(i1, j1, i2, j2):
    """
    Retourne True si la case (i2, j2) est accessible depuis (i1, j1)
    selon les règles de connectivité de Fanorona Telo :
      - Mouvements horizontaux/verticaux autorisés.
      - Mouvements diagonaux autorisés UNIQUEMENT si le point de départ est sur une diagonale principale et le mouvement reste sur cette diagonale.
    """
    x1, y1 = board_coords(i1, j1)
    x2, y2 = board_coords(i2, j2)
    dx = x2 - x1
    dy = y2 - y1
    if (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        return True
    if (dx, dy) in [(-1, -1), (1, 1), (-1, 1), (1, -1)]:
        if (x1 == y1) or (x1 == -y1):
            if (x1 == y1 and x2 == y2) or (x1 == -y1 and x2 == -y2):
                return True
    return False

def check_win_state(board, player):
    """
    Retourne True si le joueur donné a aligné trois pièces (ligne, colonne ou diagonale)
    """
    for i in range(ROWS):
        if board[i][0] == board[i][1] == board[i][2] == player:
            return True
    for j in range(COLS):
        if board[0][j] == board[1][j] == board[2][j] == player:
            return True
    if board[0][0] == board[1][1] == board[2][2] == player:
        return True
    if board[0][2] == board[1][1] == board[2][0] == player:
        return True
    return False

# ---------------------------
# Classe Node : représentation d'un état de Fanorona Telo
# ---------------------------
class Node:
    def __init__(self, board=None, phase="placement", pieces_placed=0, current_player='B'):
        if board is None:
            self.board = [[None for _ in range(COLS)] for _ in range(ROWS)]
        else:
            self.board = board
        self.phase = phase
        self.pieces_placed = pieces_placed
        self.current_player = current_player

    def copy(self):
        return Node(board=[row[:] for row in self.board],
                    phase=self.phase,
                    pieces_placed=self.pieces_placed,
                    current_player=self.current_player)

    def get_successors(self):
        """
        Génère et retourne une liste de Node successeurs (états suivants)
        en fonction de la phase du jeu.
        """
        successors = []
        if self.phase == "placement":
            for i in range(ROWS):
                for j in range(COLS):
                    if self.board[i][j] is None:
                        child = self.copy()
                        child.board[i][j] = self.current_player
                        child.pieces_placed += 1
                        if child.pieces_placed >= 6:
                            child.phase = "movement"
                        child.current_player = 'W' if self.current_player == 'B' else 'B'
                        successors.append(child)
        elif self.phase == "movement":
            for i in range(ROWS):
                for j in range(COLS):
                    if self.board[i][j] == self.current_player:
                        for ni in range(ROWS):
                            for nj in range(COLS):
                                if self.board[ni][nj] is None and is_adjacent(i, j, ni, nj):
                                    child = self.copy()
                                    child.board[ni][nj] = child.board[i][j]
                                    child.board[i][j] = None
                                    child.current_player = 'W' if self.current_player == 'B' else 'B'
                                    successors.append(child)
        return successors

    def is_terminal(self):
        """
        Retourne le gagnant ('W' ou 'B') si l'état est terminal,
        sinon retourne None.
        """
        if check_win_state(self.board, 'W'):
            return 'W'
        if check_win_state(self.board, 'B'):
            return 'B'
        return None

    def evaluate(self):
        """
        Retourne une évaluation du Node du point de vue des Blancs:
          +1 si victoire des Rouge, -1 si victoire des Bleu, 0 sinon.
        """
        winner = self.is_terminal()
        if winner == 'W':
            return 1
        elif winner == 'B':
            return -1
        return 0

# ---------------------------
# Moteurs de recherche : minimax et alpha-beta
# ---------------------------
def minimax(node, depth, maximizingPlayer):
    if depth == 0 or node.is_terminal() is not None:
        return node.evaluate(), None
    best_move = None
    if maximizingPlayer:
        maxEval = -float('inf')
        for child in node.get_successors():
            eval, _ = minimax(child, depth - 1, False)
            if eval > maxEval:
                maxEval = eval
                best_move = child
        return maxEval, best_move
    else:
        minEval = float('inf')
        for child in node.get_successors():
            eval, _ = minimax(child, depth - 1, True)
            if eval < minEval:
                minEval = eval
                best_move = child
        return minEval, best_move

def alphabeta(node, depth, alpha, beta, maximizingPlayer):
    if depth == 0 or node.is_terminal() is not None:
        return node.evaluate(), None
    best_move = None
    if maximizingPlayer:
        maxEval = -float('inf')
        for child in node.get_successors():
            eval, _ = alphabeta(child, depth - 1, alpha, beta, False)
            if eval > maxEval:
                maxEval = eval
                best_move = child
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval, best_move
    else:
        minEval = float('inf')
        for child in node.get_successors():
            eval, _ = alphabeta(child, depth - 1, alpha, beta, True)
            if eval < minEval:
                minEval = eval
                best_move = child
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval, best_move

# ---------------------------
# Interface graphique et popup
# ---------------------------
def get_cell(pos):
    """
    Convertit une position en pixels en indices de grille (i, j).
    """
    x, y = pos
    i = y // CELL_SIZE
    j = x // CELL_SIZE
    return i, j
def load_background():
    try:
        bg = pygame.image.load("wood_bg.jpg")
        bg = pygame.transform.scale(bg, (WIDTH, HEIGHT))
    except Exception as e:
        print("wood_bg.jpg non trouvée, utilisation d'une couleur approximative.")
        bg = None
    return bg

def draw_grid(win):
    """Trace les lignes du plateau avec des intersections circulaires pour accueillir les pièces."""
    for i in range(ROWS):
        for j in range(COLS):
            cx, cy = j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2

            # 🔹 Dessin des lignes entre les intersections connectées
            for k in range(ROWS):
                for l in range(COLS):
                    if (k, l) <= (i, j):
                        continue
                    if is_adjacent(i, j, k, l):
                        nx, ny = l * CELL_SIZE + CELL_SIZE // 2, k * CELL_SIZE + CELL_SIZE // 2
                        pygame.draw.line(win, GRID_COLOR, (cx, cy), (nx, ny), 3)

            # 🔹 Ajout des cercles aux intersections
            pygame.draw.circle(win, GRID_COLOR, (cx, cy), CELL_SIZE // 6, 3)  # Cercle vide à l'intersection



def draw_board_state(win, state, selected, bg=None):
    """Affiche l'état du plateau avec le nouveau design."""
    if bg:
        win.blit(bg, (0, 0))  # Si une image de fond est disponible
    else:
        win.fill(BG_COLOR)  # Sinon, fond coloré
    
    draw_grid(win)

    for i in range(ROWS):
        for j in range(COLS):
            cx, cy = j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2
            if state.board[i][j] == 'B':
                pygame.draw.circle(win, BLUE_PIECE, (cx, cy), CELL_SIZE//5)
            elif state.board[i][j] == 'W':
                pygame.draw.circle(win, RED_PIECE, (cx, cy), CELL_SIZE//5)

    if selected:
        i, j = selected
        pygame.draw.rect(win, SELECTED_COLOR, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE), 3)

    pygame.display.update()




def show_popup(win, message):
    popup_width, popup_height = 400, 200
    popup_rect = pygame.Rect((WIDTH - popup_width)//2, (HEIGHT - popup_height)//2, popup_width, popup_height)
    button_rect = pygame.Rect((WIDTH - 150)//2, popup_rect.top + 130, 150, 40)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(pygame.mouse.get_pos()):
                    return  # Quitte la popup
        popup_surface = pygame.Surface((popup_width, popup_height), pygame.SRCALPHA)
        popup_surface.fill(POPUP_BG_COLOR)
        win.blit(popup_surface, popup_rect.topleft)
        text_surf = POPUP_FONT.render(message, True, (255, 255, 255))
        win.blit(text_surf, (popup_rect.centerx - text_surf.get_width()//2, popup_rect.top + 40))
        pygame.draw.rect(win, BUTTON_COLOR, button_rect)
        btn_text = FONT.render("Restart", True, (255,255,255))
        win.blit(btn_text, (button_rect.centerx - btn_text.get_width()//2, button_rect.centery - btn_text.get_height()//2))
        pygame.display.update()

# ---------------------------
# Expérimentation de performance
# ---------------------------
def compare_algorithms(test_node, depth, iterations=20):
    """
    Compare le temps moyen d'exécution de minimax classique et de alphabeta sur le même état.
    Affiche une barre de progression dans le terminal.
    Retourne (avg_time_minimax, avg_time_alphabeta)
    """
    times_minimax = []
    times_alphabeta = []
    for i in range(iterations):
        start = time.time()
        minimax(test_node, depth, True)
        times_minimax.append(time.time() - start)
        
        start = time.time()
        alphabeta(test_node, depth, -float('inf'), float('inf'), True)
        times_alphabeta.append(time.time() - start)
        
        # Affichage de la progression
        progress = (i + 1) / iterations * 100
        print(f"Comparaison des algorithmes : {progress:.0f}% complet", end="\r")
    print()  # Pour passer à la ligne suivante après la boucle
    avg_minimax = sum(times_minimax) / iterations
    avg_alphabeta = sum(times_alphabeta) / iterations
    return avg_minimax, avg_alphabeta

# ---------------------------
# Génération de dataset terminal (positions gagnantes et perdantes pour les blancs)
# ---------------------------
def generate_dataset(n):
    """
    Génère n configurations terminales gagnantes pour les blancs (label +1)
    et n configurations terminales perdantes pour les blancs (label -1).
    Chaque configuration est représentée comme un vecteur de 9 éléments (en ordre ligne-par-ligne)
    avec : Blanc = 1, Noir = -1, vide = 0.
    Affiche une barre de progression dans le terminal.
    Retourne X (features) et y (labels).
    """
    dataset_X = []
    dataset_y = []
    attempts = 0
    required = 2 * n
    while len(dataset_X) < required and attempts < 10000:
        attempts += 1
        positions = list(range(9))
        random.shuffle(positions)
        white_positions = sorted(positions[:3])
        black_positions = sorted(positions[3:6])
        # Créer le board 3x3
        board = [[None for _ in range(COLS)] for _ in range(ROWS)]
        for pos in white_positions:
            i, j = divmod(pos, COLS)
            board[i][j] = 'W'
        for pos in black_positions:
            i, j = divmod(pos, COLS)
            board[i][j] = 'B'
        # On teste le résultat terminal
        if check_win_state(board, 'W'):
            # Configuration gagnante pour les rouges
            vector = [1 if cell == 'W' else -1 if cell == 'B' else 0 for row in board for cell in row]
            dataset_X.append(vector)
            dataset_y.append(1)
        elif check_win_state(board, 'B'):
            # Configuration perdante pour les blancs (victoire des bleu)
            vector = [1 if cell == 'W' else -1 if cell == 'B' else 0 for row in board for cell in row]
            dataset_X.append(vector)
            dataset_y.append(-1)
        
        # Mise à jour de la barre de progression
        progress = len(dataset_X) / required * 100
        print(f"Generation du dataset : {progress:.0f}% complet", end="\r")
    print()  # Passage à la ligne après la boucle
    print(f"Configurations générées: {len(dataset_X)} (après {attempts} tentatives)")
    return np.array(dataset_X), np.array(dataset_y)

def train_score_model(X, y):
    """
    Utilise la régression linéaire pour prédire le score d'une position (du point de vue des blancs).
    Utilise 80% des données pour l'entraînement et 20% pour le test.
    Affiche les performances (MSE, R2).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Performance du modèle de prédiction du score (du point de vue des blancs) :")
    print(f"MSE : {mse:.4f}, R² : {r2:.4f}")
    return model

# ---------------------------
# Partie interactive avec interface Pygame et moteur IA
# ---------------------------
def game_interface():
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Fanorona Telo - AlphaBeta & Minimax")
    clock = pygame.time.Clock()
    bg = None  # Vous pouvez charger une image bois ici avec load_background()
    bg = load_background()
    running = True

    def init_state():
        return Node()  # État initial par défaut (phase placement, plateau vide, joueur 'B')
    
    state = init_state()
    selected = None
    human_player = 'B'  # Humain joue Noir
    ai_player = 'W'     # IA joue Blanc

    while running:
        clock.tick(30)
        draw_board_state(win, state, selected, bg)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False; break
            if state.current_player == human_player and event.type == pygame.MOUSEBUTTONDOWN:
                i, j = get_cell(pygame.mouse.get_pos())
                if not (0 <= i < ROWS and 0 <= j < COLS):
                    continue
                if state.phase == "placement":
                    if state.board[i][j] is None:
                        state.board[i][j] = human_player
                        state.pieces_placed += 1
                        if check_win_state(state.board, human_player):
                            draw_board_state(win, state, selected, bg)
                            show_popup(win, "Victoire des Bleu en placement !")
                            state = init_state(); continue
                        state.current_player = ai_player
                        if state.pieces_placed == 6:
                            state.phase = "movement"
                elif state.phase == "movement":
                    if selected is None:
                        if state.board[i][j] == human_player:
                            selected = (i, j)
                    else:
                        si, sj = selected
                        if state.board[i][j] is None and is_adjacent(si, sj, i, j):
                            state.board[i][j] = state.board[si][sj]
                            state.board[si][sj] = None
                            if check_win_state(state.board, human_player):
                                draw_board_state(win, state, selected, bg)
                                show_popup(win, "Victoire des Bleu en déplacement !")
                                state = init_state(); selected = None; continue
                            state.current_player = ai_player
                        selected = None
        if state.current_player == ai_player and running:
            depth = 9
            # On utilise alphabeta ici ; pour tester minimax, il suffit de remplacer par minimax(state, depth, True)
            _, best_child = alphabeta(state, depth, -float('inf'), float('inf'), True)
            if best_child is not None:
                state = best_child
                if check_win_state(state.board, ai_player):
                    draw_board_state(win, state, selected, bg)
                    show_popup(win, "Victoire des Rouges !")
                    state = init_state()
            else:
                print("Aucun coup possible pour l'IA, match nul.")
                show_popup(win, "Match nul")
                state = init_state()
    pygame.quit()
    sys.exit()

# ---------------------------
# Partie expérimentale : comparaison de performances et apprentissage
# ---------------------------
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def run_experiments(depth=9, iterations=50, dataset_size=500):
    """
    Expérimente l'efficacité des algorithmes et entraîne un modèle d'évaluation.
    - Compare les performances de Minimax et Alpha-Beta.
    - Génère un dataset de positions gagnantes/perdantes et entraîne un modèle de prédiction.
    - Affiche un graphique des performances.
    """
    print("🔍 Comparaison des algorithmes Minimax et Alpha-Beta...\n")
    
    # Comparaison des algorithmes
    init_node = Node()
    avg_minimax, avg_alphabeta = compare_algorithms(init_node, depth, iterations)
    
    # Calcul du gain d'efficacité
    gain = (1 - avg_alphabeta / avg_minimax) * 100 if avg_minimax > 0 else 0

    # 📊 Affichage sous forme de tableau
    results_df = pd.DataFrame({
        "Algorithme": ["Minimax", "Alpha-Beta"],
        "Temps moyen (s)": [avg_minimax, avg_alphabeta]
    })
    print(results_df)
    print(f"\n🔺 Gain d'efficacité d'Alpha-Beta : {gain:.1f}%\n")

    # 📈 Affichage des performances sous forme de graphique
    plt.figure(figsize=(7, 5))
    plt.bar(["Minimax", "Alpha-Beta"], [avg_minimax, avg_alphabeta], color=["red", "green"])
    plt.ylabel("Temps moyen (secondes)")
    plt.title("Comparaison des performances Minimax vs Alpha-Beta")
    plt.show()

    # 🔄 Génération du dataset et entraînement du modèle
    print("\n📊 Génération du dataset et entraînement du modèle...")
    X, y = generate_dataset(dataset_size)
    model = train_score_model(X, y)

    # 💾 Sauvegarde du modèle
    with open("trained_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("✅ Modèle de prédiction sauvegardé sous 'trained_model.pkl'.")


    
def main():
    print("1) Jouer")
    print("2) Exécuter les expériences")
    choice = input("Votre choix : ")
    if choice.strip() == "1":
        game_interface()
    else:
        run_experiments()

if __name__ == '__main__':
    main()
