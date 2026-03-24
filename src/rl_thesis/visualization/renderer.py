"""
Minimalist pygame renderer for the survival environment.

Provides a simple visual representation using colored dots
for entities on a plain background.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple
import numpy as np
import pygame
from rl_thesis.environment.world import WorldState

if TYPE_CHECKING:
    from rl_thesis.config.config import VisualizationConfig


class Renderer:
    """
    pygame renderer for the survival world.
    """
    
    def __init__(
        self,
        config: VisualizationConfig,
        world_width: int,
        world_height: int,
    ):
        self.config = config
        self.world_width = world_width
        self.world_height = world_height
        
        # Calculate window size
        self.game_width = self.world_width * self.config.cell_size
        self.game_height = self.world_height * self.config.cell_size
        self.window_width = self.game_width
        self.window_height = self.game_height + self.config.hud_height
        
        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("Survival RL Environment")
        
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height)
        )
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Pre-render background surface
        self._bg_surface: Optional[pygame.Surface] = None
        
        # Running state
        self.is_open = True
    
    def _create_background_surface(self) -> pygame.Surface:
        """Create a plain background surface."""
        surface = pygame.Surface((self.game_width, self.game_height))
        surface.fill((144, 238, 144))  # Light green
        return surface
    
    def render(
        self,
        state: WorldState,
        metrics: Optional[dict] = None,
    ) -> bool:
        """
        Render the current world state.
        
        Args:
            state: Current world state snapshot
            metrics: Optional metrics dict for the side panel
            
        Returns:
            False if window was closed, True otherwise
        """
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_open = False
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.is_open = False
                    return False
        
        # Clear screen
        self.screen.fill(self.config.background_color)
        
        # Draw background
        if self._bg_surface is None:
            self._bg_surface = self._create_background_surface()
        self.screen.blit(self._bg_surface, (0, 0))
        
        # Draw shelters
        for x, y in state.shelters:
            self._draw_entity(x, y, self.config.shelter_color, size_ratio=0.9)
        
        # Draw food
        for x, y in state.food:
            self._draw_entity(x, y, self.config.food_color, size_ratio=0.5)
        
        # Draw enemies
        for x, y in state.enemies:
            self._draw_entity(x, y, self.config.enemy_color, size_ratio=0.6)
        
        # Draw agent
        agent_x, agent_y = state.agent_position
        self._draw_entity(
            agent_x, agent_y, 
            self.config.agent_color, 
            size_ratio=0.8
        )
        
        # Draw HUD
        self._draw_hud(state)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.config.fps)
        
        return True
    
    def _draw_entity(
        self,
        x: int,
        y: int,
        color: Tuple[int, int, int],
        size_ratio: float = 0.6
    ) -> None:
        """Draw an entity as a colored circle."""
        cell_size = self.config.cell_size
        radius = int(cell_size * size_ratio / 2)
        center_x = x * cell_size + cell_size // 2
        center_y = y * cell_size + cell_size // 2
        
        pygame.draw.circle(self.screen, color, (center_x, center_y), radius)
    
    def _draw_hud(self, state: WorldState) -> None:
        """Draw the heads-up display with health and hunger bars."""
        hud_y = self.game_height
        bar_width = 200
        bar_height = 20
        margin = 10
        
        # Background
        hud_rect = pygame.Rect(0, hud_y, self.game_width, self.config.hud_height)
        pygame.draw.rect(self.screen, (40, 40, 40), hud_rect)
        
        # Health bar
        health_ratio = state.agent_health / state.agent_max_health
        self._draw_bar(
            margin, hud_y + margin,
            bar_width, bar_height,
            health_ratio,
            self.config.health_bar_color,
            "Health"
        )
        
        # Hunger bar
        hunger_ratio = state.agent_hunger / state.agent_max_hunger
        self._draw_bar(
            margin + bar_width + 40, hud_y + margin,
            bar_width, bar_height,
            hunger_ratio,
            self.config.hunger_bar_color,
            "Hunger"
        )
        
        # Ticks survived
        ticks_text = self.font.render(
            f"Ticks: {state.ticks}", True, (255, 255, 255)
        )
        self.screen.blit(ticks_text, (margin + 2 * bar_width + 80, hud_y + margin))
        
        # Shelter indicator
        if state.agent_in_shelter:
            shelter_text = self.font.render(
                "🏠 IN SHELTER", True, (100, 200, 100)
            )
            self.screen.blit(shelter_text, (margin + 2 * bar_width + 80, hud_y + margin + 25))
    
    def _draw_bar(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        ratio: float,
        color: Tuple[int, int, int],
        label: str
    ) -> None:
        """Draw a progress bar with label."""
        # Background
        bg_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (60, 60, 60), bg_rect)
        
        # Fill
        fill_width = int(width * max(0, min(1, ratio)))
        fill_rect = pygame.Rect(x, y, fill_width, height)
        pygame.draw.rect(self.screen, color, fill_rect)
        
        # Border
        pygame.draw.rect(self.screen, (100, 100, 100), bg_rect, 2)
        
        # Label
        label_text = self.small_font.render(
            f"{label}: {ratio*100:.0f}%", True, (255, 255, 255)
        )
        self.screen.blit(label_text, (x + 5, y + 3))

    def close(self) -> None:
        """Close the renderer and pygame."""
        pygame.quit()
        self.is_open = False
    
    def is_running(self) -> bool:
        """Check if the window is still open."""
        return self.is_open


class HeadlessRenderer:
    """
    Dummy renderer for headless operation.
    
    Provides the same interface as Renderer but does nothing,
    allowing code to work without pygame installed.
    """
    
    def __init__(self, *args, **kwargs):
        pass
    
    def render(self, *args, **kwargs) -> bool:
        return True
    
    def close(self) -> None:
        pass
    
    def is_running(self) -> bool:
        return True


def create_renderer(
    config: VisualizationConfig,
    world_width: int,
    world_height: int,
    headless: bool = False,
) -> Renderer | HeadlessRenderer:
    if headless:
        return HeadlessRenderer()
    return Renderer(config, world_width, world_height)
