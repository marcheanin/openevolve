# Category Distribution in Stratified Splits (All Categories)

This document shows the distribution of Amazon product categories across the `train`, `validation`, and `test` splits when using the stratified sampling approach (`stratify_users: true`).

## Configuration
- **max_train_users:** 25
- **max_val_users:** 15
- **max_reviews_per_user:** 15

## Train Split
**Total Reviews:** 375 | **Categories Covered:** 20 / 26

| Rank | Category ID | Category Name | Reviews |
|---|---|---|---|
| 1 | 0 | Books | 196 |
| 2 | 1 | Kindle_Store | 24 |
| 3 | 24 | Home_and_Kitchen | 23 |
| 4 | 21 | Grocery_and_Gourmet_Food | 19 |
| 5 | 7 | Tools_and_Home_Improvement | 16 |
| 6 | 18 | Movies_and_TV | 15 |
| 7 | 5 | Toys_and_Games | 14 |
| 8 | 20 | Electronics | 13 |
| 9 | 16 | Clothing_Shoes_and_Jewelry | 12 |
| 10 | 12 | CDs_and_Vinyl | 10 |
| 11 | 11 | Pet_Supplies | 9 |
| 12 | 10 | Prime_Pantry | 5 |
| 13 | 6 | Automotive | 4 |
| 14 | 8 | Sports_and_Outdoors | 4 |
| 15 | 14 | Office_Products | 3 |
| 16 | 13 | Patio_Lawn_and_Garden | 3 |
| 17 | 15 | Cell_Phones_and_Accessories | 2 |
| 18 | 4 | Arts_Crafts_and_Sewing | 1 |
| 19 | 25 | Industrial_and_Scientific | 1 |
| 20 | 2 | Video_Games | 1 |

## Validation Split
**Total Reviews:** 225 | **Categories Covered:** 13 / 26

| Rank | Category ID | Category Name | Reviews |
|---|---|---|---|
| 1 | 0 | Books | 110 |
| 2 | 1 | Kindle_Store | 26 |
| 3 | 18 | Movies_and_TV | 26 |
| 4 | 24 | Home_and_Kitchen | 14 |
| 5 | 11 | Pet_Supplies | 12 |
| 6 | 21 | Grocery_and_Gourmet_Food | 10 |
| 7 | 20 | Electronics | 8 |
| 8 | 12 | CDs_and_Vinyl | 7 |
| 9 | 16 | Clothing_Shoes_and_Jewelry | 5 |
| 10 | 8 | Sports_and_Outdoors | 3 |
| 11 | 14 | Office_Products | 2 |
| 12 | 25 | Industrial_and_Scientific | 1 |
| 13 | 7 | Tools_and_Home_Improvement | 1 |

## Test Split
**Total Reviews:** 225 | **Categories Covered:** 20 / 26

| Rank | Category ID | Category Name | Reviews |
|---|---|---|---|
| 1 | 0 | Books | 97 |
| 2 | 1 | Kindle_Store | 21 |
| 3 | 24 | Home_and_Kitchen | 18 |
| 4 | 12 | CDs_and_Vinyl | 15 |
| 5 | 16 | Clothing_Shoes_and_Jewelry | 11 |
| 6 | 21 | Grocery_and_Gourmet_Food | 11 |
| 7 | 14 | Office_Products | 8 |
| 8 | 20 | Electronics | 7 |
| 9 | 11 | Pet_Supplies | 7 |
| 10 | 7 | Tools_and_Home_Improvement | 7 |
| 11 | 18 | Movies_and_TV | 6 |
| 12 | 13 | Patio_Lawn_and_Garden | 3 |
| 13 | 8 | Sports_and_Outdoors | 3 |
| 14 | 4 | Arts_Crafts_and_Sewing | 2 |
| 15 | 6 | Automotive | 2 |
| 16 | 15 | Cell_Phones_and_Accessories | 2 |
| 17 | 5 | Toys_and_Games | 2 |
| 18 | 23 | Luxury_Beauty | 1 |
| 19 | 10 | Prime_Pantry | 1 |
| 20 | 2 | Video_Games | 1 |

## Combined (Train / Validation / Test)

| Category ID | Category Name | Train | Validation | Test |
|---:|---|---:|---:|---:|
| 0 | Books | 196 | 110 | 97 |
| 1 | Kindle_Store | 24 | 26 | 21 |
| 2 | Video_Games | 1 | 0 | 1 |
| 4 | Arts_Crafts_and_Sewing | 1 | 0 | 2 |
| 5 | Toys_and_Games | 14 | 0 | 2 |
| 6 | Automotive | 4 | 0 | 2 |
| 7 | Tools_and_Home_Improvement | 16 | 1 | 7 |
| 8 | Sports_and_Outdoors | 4 | 3 | 3 |
| 10 | Prime_Pantry | 5 | 0 | 1 |
| 11 | Pet_Supplies | 9 | 12 | 7 |
| 12 | CDs_and_Vinyl | 10 | 7 | 15 |
| 13 | Patio_Lawn_and_Garden | 3 | 0 | 3 |
| 14 | Office_Products | 3 | 2 | 8 |
| 15 | Cell_Phones_and_Accessories | 2 | 0 | 2 |
| 16 | Clothing_Shoes_and_Jewelry | 12 | 5 | 11 |
| 18 | Movies_and_TV | 15 | 26 | 6 |
| 20 | Electronics | 13 | 8 | 7 |
| 21 | Grocery_and_Gourmet_Food | 19 | 10 | 11 |
| 23 | Luxury_Beauty | 0 | 0 | 1 |
| 24 | Home_and_Kitchen | 23 | 14 | 18 |
| 25 | Industrial_and_Scientific | 1 | 1 | 0 |

**Totals:** Train=375, Validation=225, Test=225
