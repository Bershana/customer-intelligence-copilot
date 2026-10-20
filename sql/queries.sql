-- Q1 WHERE then HAVING: completed orders above INR 500, customers whose total of those orders exceeds INR 8000
SELECT customer_id, ROUND(SUM(order_amount), 2) AS total_amount
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- Q2 three-table join: completed revenue by city and product category
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Q3 scalar subquery: products priced above the average catalogue price
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4 LEFT JOIN: customers who have no completed order
SELECT c.customer_id
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5 CTE: customers whose completed spend is above the average customer spend
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT AVG(total_spend) AS avg_total
    FROM customer_spend
)
SELECT cs.customer_id, ROUND(cs.total_spend, 2) AS total_spend
FROM customer_spend cs
CROSS JOIN avg_spend a
WHERE cs.total_spend > a.avg_total
ORDER BY cs.total_spend DESC;

-- Q6 ranking window functions by city
WITH spend AS (
    SELECT c.customer_id, c.city, SUM(o.order_amount) AS total_spend
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT customer_id,
       city,
       ROUND(total_spend, 2) AS total_spend,
       RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rank_num,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rank_num,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM spend
ORDER BY city, rank_num, row_num;

-- Q7 running total of daily completed revenue
WITH daily AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT day,
       ROUND(revenue, 2) AS revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily
ORDER BY day;
