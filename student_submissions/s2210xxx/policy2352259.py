import numpy as np
from policy import Policy

class Policy2352259(Policy):
    def __init__(self):
        super().__init__()
        self.name = "Enhanced Best-Fit Policy"
        
    def _get_valid_sizes(self, prod_size, stock_size):
        """Get valid orientations that fit within stock"""
        sizes = [prod_size]
        if prod_size[0] != prod_size[1]:  # If not square
            sizes.append((prod_size[1], prod_size[0]))  # Add rotated version
            
        return [size for size in sizes 
                if size[0] <= stock_size[0] and size[1] <= stock_size[1]]
    
    def _score_position(self, stock, pos, size):
        """Score placement position based on multiple factors"""
        stock_h, stock_w = stock.shape
        pos_h, pos_w = pos
        size_h, size_w = size
        
        # Edge alignment bonus
        edge_score = 0
        if pos_h == 0 or pos_h + size_h == stock_h:
            edge_score += 1
        if pos_w == 0 or pos_w + size_w == stock_w:
            edge_score += 1
            
        # Corner proximity
        corners = [(0,0), (0,stock_w-1), (stock_h-1,0), (stock_h-1,stock_w-1)]
        corner_dist = min(abs(pos_h-ch) + abs(pos_w-cw) for ch, cw in corners)
        
        # Calculate adjacent filled cells bonus
        adjacent_filled = 0
        if pos_h > 0:  # Check above
            adjacent_filled += np.sum(stock[pos_h-1, pos_w:pos_w+size_w] >= 0)
        if pos_h + size_h < stock_h:  # Check below
            adjacent_filled += np.sum(stock[pos_h+size_h, pos_w:pos_w+size_w] >= 0)
        if pos_w > 0:  # Check left
            adjacent_filled += np.sum(stock[pos_h:pos_h+size_h, pos_w-1] >= 0)
        if pos_w + size_w < stock_w:  # Check right
            adjacent_filled += np.sum(stock[pos_h:pos_h+size_h, pos_w+size_w] >= 0)
            
        return {
            'edge_score': edge_score,
            'corner_dist': corner_dist,
            'adjacent_filled': adjacent_filled
        }
    
    def _find_best_position(self, stock, size):
        """Find optimal position for placement"""
        stock_h, stock_w = stock.shape
        best_pos = None
        best_score = float('-inf')
        
        for i in range(stock_h - size[0] + 1):
            for j in range(stock_w - size[1] + 1):
                if not self._can_place_(stock, (i,j), size):
                    continue
                    
                scores = self._score_position(stock, (i,j), size)
                total_score = (scores['edge_score'] * 10 + 
                             scores['adjacent_filled'] * 5 - 
                             scores['corner_dist'] * 0.1)
                
                if total_score > best_score:
                    best_score = total_score
                    best_pos = (i,j)
                    
        return best_pos, best_score
    
    def get_action(self, observation, info):
        """Get next cutting action using enhanced best-fit strategy"""
        stocks = observation["stocks"]
        products = observation["products"]
        
        best_action = None
        best_total_score = float('-inf')
        
        # Try each product with remaining quantity
        for prod in products:
            if prod["quantity"] <= 0:
                continue
                
            # Try each stock
            for stock_idx, stock in enumerate(stocks):
                stock_size = self._get_stock_size_(stock)
                
                # Try each valid orientation
                valid_sizes = self._get_valid_sizes(prod["size"], stock_size)
                for size in valid_sizes:
                    pos, score = self._find_best_position(stock, size)
                    
                    if pos is not None and score > best_total_score:
                        best_total_score = score
                        best_action = {
                            "stock_idx": stock_idx,
                            "size": size,
                            "position": pos
                        }
        
        return best_action