# Coding Task Graph Theory Group 4

|Name|NRP|
|--|--|
|Naura Rizky Ameira|	5025251001|
|Farras Al Ghifari|	5025251006|
|Nisrina Zahralila|	5025251013|
|Safiya Nadira Az-zahra|	5025251268|

## PREREQUISITES

To run this project, ensure you have the following installed:
* **Python 3.x** (Python 3.8 or higher recommended)
* `tsp.py`
* `cpp.py`
* `Chessknight.py`

### Dependencies
No external libraries or third-party packages are required as it only uses Python's built-in standard libraries.

---

## Travelling Salesman Problem

### How to run

### Sample Test Case
---

## Chinese Postman Problem

### How to run
### Sample Test Case
---

## Knight's Tour

### How to run

### Input format

Line 1: width and height of chessboard, separated by space e.g. 5 5

Line 2: starting column and row coordinates of knight, separated by space e.g. 2 2

### Expected Output

If a path is found, the program prints `path found` followed by the exact ordered sequence of coordinates visited from start to finish.

If the knight gets trapped or a full tour is impossible, the program prints `path not found` followed by a list of all unreached coordinates remaining on the board.

### Sample Test Case

### Valid testcase

Input:

6 6

0 0

Output:

path found
0,0
2,1
4,2
5,4
3,5
1,4
0,2
2,3
4,4
2,5
0,4
1,2
2,4
0,5
1,3
0,1
2,0
4,1
5,3
4,5
3,3
5,2
4,0
3,2
1,1
0,3
1,5
3,4
5,5
4,3
5,1
3,0
2,2
1,0
3,1
5,0

### Invalid testcase

Input:

4 4

0 0

Output:
