
import { Pattern } from '../types';

export const CODING_PATTERNS: Pattern[] = [
  {
    id: 1,
    name: 'Sliding Window',
    whenToUse: 'Problems involving a contiguous subarray, substring, or a specific-sized window over a linear data structure like an array or string.',
    approach: 'Maintain a "window" (a sub-array or sub-string) and slide it from left to right. Expand the window by adding elements from the right, and shrink it by removing elements from the left based on certain conditions.',
    example: 'Find the maximum sum of any contiguous subarray of size ‘k’',
    problems: [
      {
        title: 'Maximum Sum Subarray of Size K',
        pseudoCode: `function maxSumSubarray(arr, k):
  maxSum = 0
  windowSum = 0
  windowStart = 0

  for windowEnd from 0 to arr.length - 1:
    windowSum += arr[windowEnd]

    if windowEnd >= k - 1:
      maxSum = max(maxSum, windowSum)
      windowSum -= arr[windowStart]
      windowStart++
      
  return maxSum`
      },
      {
        title: 'Longest Substring with K Distinct Characters',
        pseudoCode: `function longestSubstringWithKDistinct(str, k):
  maxLength = 0
  windowStart = 0
  charFrequency = {}

  for windowEnd from 0 to str.length - 1:
    rightChar = str[windowEnd]
    if rightChar not in charFrequency:
      charFrequency[rightChar] = 0
    charFrequency[rightChar] += 1

    while length of charFrequency > k:
      leftChar = str[windowStart]
      charFrequency[leftChar] -= 1
      if charFrequency[leftChar] == 0:
        delete charFrequency[leftChar]
      windowStart++

    maxLength = max(maxLength, windowEnd - windowStart + 1)
    
  return maxLength`
      },
      {
        title: 'String Anagrams',
        pseudoCode: `function findStringAnagrams(str, pattern):
  resultIndices = []
  windowStart = 0
  matched = 0
  charFrequency = {}

  for char in pattern:
    if char not in charFrequency:
      charFrequency[char] = 0
    charFrequency[char] += 1
  
  for windowEnd from 0 to str.length - 1:
    rightChar = str[windowEnd]
    if rightChar in charFrequency:
      charFrequency[rightChar] -= 1
      if charFrequency[rightChar] == 0:
        matched += 1
    
    if matched == length of charFrequency:
      resultIndices.add(windowStart)

    if windowEnd >= pattern.length - 1:
      leftChar = str[windowStart]
      windowStart++
      if leftChar in charFrequency:
        if charFrequency[leftChar] == 0:
          matched -= 1
        charFrequency[leftChar] += 1

  return resultIndices`
      }
    ]
  },
  {
    id: 2,
    name: 'Two Pointers',
    whenToUse: 'Problems involving sorted arrays or linked lists where you need to find a pair, a triplet, or a sub-array that satisfies certain constraints.',
    approach: 'Use two pointers, one starting from the beginning (left) and one from the end (right). Move them towards each other based on the conditions, reducing the search space.',
    example: 'Find a pair in a sorted array whose sum is equal to a given target.',
    problems: [
      {
        title: 'Pair with Target Sum (Sorted Array)',
        pseudoCode: `function pairWithTargetSum(arr, targetSum):
  left = 0
  right = arr.length - 1

  while left < right:
    currentSum = arr[left] + arr[right]
    if currentSum == targetSum:
      return [left, right]
    
    if targetSum > currentSum:
      left++ // we need a pair with a bigger sum
    else:
      right-- // we need a pair with a smaller sum

  return [-1, -1]`
      },
      {
        title: 'Remove Duplicates from Sorted Array',
        pseudoCode: `function removeDuplicates(arr):
  if arr is empty: return 0
  
  nextNonDuplicate = 1
  for i from 1 to arr.length - 1:
    if arr[nextNonDuplicate - 1] != arr[i]:
      arr[nextNonDuplicate] = arr[i]
      nextNonDuplicate++
      
  return nextNonDuplicate`
      },
      {
        title: 'Squaring a Sorted Array',
        pseudoCode: `function makeSquares(arr):
  n = arr.length
  squares = new Array(n)
  highestSquareIdx = n - 1
  left = 0
  right = n - 1

  while left <= right:
    leftSquare = arr[left] * arr[left]
    rightSquare = arr[right] * arr[right]
    if leftSquare > rightSquare:
      squares[highestSquareIdx] = leftSquare
      left++
    else:
      squares[highestSquareIdx] = rightSquare
      right--
    highestSquareIdx--
  
  return squares`
      }
    ]
  },
  {
    id: 3,
    name: 'Fast & Slow Pointers',
    whenToUse: 'Problems involving cycles in linked lists or arrays, or finding the middle/k-th element of a linked list.',
    approach: 'Use two pointers, a slow pointer that moves one step at a time, and a fast pointer that moves two steps at a time. Their relative positions help solve the problem.',
    example: 'Detect a cycle in a linked list.',
    problems: [
      {
        title: 'Linked List Cycle Detection',
        pseudoCode: `function hasCycle(head):
  slow = head
  fast = head

  while fast is not null and fast.next is not null:
    fast = fast.next.next
    slow = slow.next
    if slow == fast:
      return true // found the cycle
      
  return false`
      },
      {
        title: 'Find the Middle of a Linked List',
        pseudoCode: `function findMiddle(head):
  slow = head
  fast = head

  while fast is not null and fast.next is not null:
    slow = slow.next
    fast = fast.next.next
    
  return slow`
      },
      {
        title: 'Happy Number',
        pseudoCode: `function findHappyNumber(num):
  slow = num
  fast = num
  
  loop:
    slow = findSquareSum(slow)
    fast = findSquareSum(findSquareSum(fast))
    if slow == fast:
      break
      
  return slow == 1

function findSquareSum(num):
  sum = 0
  while num > 0:
    digit = num % 10
    sum += digit * digit
    num = floor(num / 10)
  return sum`
      }
    ]
  },
   {
    id: 4,
    name: 'DFS (Depth-First Search)',
    whenToUse: 'Problems involving traversing or searching tree or graph structures. Good for pathfinding, connectivity, or when exploring one branch completely before moving to the next is beneficial.',
    approach: 'Start at a root node and explore as far as possible along each branch before backtracking. Typically implemented recursively or iteratively using a stack.',
    example: 'Find if a path exists between two nodes in a graph.',
    problems: [
      {
        title: 'Path Sum in a Binary Tree',
        pseudoCode: `function hasPathSum(root, sum):
  if root is null:
    return false

  // if it's a leaf node and value equals sum, we found a path
  if root.value == sum and root.left is null and root.right is null:
    return true

  // recursively call to traverse the left and right sub-tree
  return hasPathSum(root.left, sum - root.value) or 
         hasPathSum(root.right, sum - root.value)`
      },
      {
        title: 'All Paths for a Sum',
        pseudoCode: `function findAllPaths(root, sum):
  allPaths = []
  findPathsRecursive(root, sum, [], allPaths)
  return allPaths

function findPathsRecursive(currentNode, sum, currentPath, allPaths):
  if currentNode is null:
    return

  currentPath.add(currentNode.value)

  if currentNode.value == sum and currentNode.left is null and currentNode.right is null:
    allPaths.add(list(currentPath))
  else:
    findPathsRecursive(currentNode.left, sum - currentNode.value, currentPath, allPaths)
    findPathsRecursive(currentNode.right, sum - currentNode.value, currentPath, allPaths)

  // backtrack
  currentPath.removeLast()`
      },
      {
        title: 'Number of Islands (Matrix)',
        pseudoCode: `function countIslands(matrix):
  if matrix is empty: return 0
  rows = matrix.length
  cols = matrix[0].length
  count = 0
  for i from 0 to rows - 1:
    for j from 0 to cols - 1:
      if matrix[i][j] == '1':
        count += 1
        dfs(matrix, i, j)
  return count

function dfs(matrix, r, c):
  if r < 0 or r >= matrix.length or c < 0 or c >= matrix[0].length or matrix[r][c] == '0':
    return
  matrix[r][c] = '0' // mark as visited
  dfs(matrix, r + 1, c)
  dfs(matrix, r - 1, c)
  dfs(matrix, r, c + 1)
  dfs(matrix, r, c - 1)`
      }
    ]
  },
  {
    id: 5,
    name: 'BFS (Breadth-First Search)',
    whenToUse: 'Traversing or searching tree/graph structures to find the shortest path between two nodes, or for level-order traversal.',
    approach: 'Start at a root node and explore all neighbor nodes at the present depth prior to moving on to nodes at the next depth level. Implemented using a queue.',
    example: 'Level order traversal of a binary tree.',
    problems: [
      {
        title: 'Binary Tree Level Order Traversal',
        pseudoCode: `function traverse(root):
  result = []
  if root is null:
    return result
  
  queue = Queue()
  queue.enqueue(root)
  while queue is not empty:
    levelSize = queue.size()
    currentLevel = []
    for i from 0 to levelSize - 1:
      currentNode = queue.dequeue()
      currentLevel.add(currentNode.value)
      if currentNode.left is not null:
        queue.enqueue(currentNode.left)
      if currentNode.right is not null:
        queue.enqueue(currentNode.right)
    result.add(currentLevel)
    
  return result`
      },
      {
        title: 'Zigzag Traversal',
        pseudoCode: `function zigzagTraverse(root):
  result = []
  if root is null: return result
  
  queue = Queue()
  queue.enqueue(root)
  leftToRight = true
  while queue is not empty:
    levelSize = queue.size()
    currentLevel = Deque()
    for i from 0 to levelSize - 1:
      currentNode = queue.dequeue()
      if leftToRight:
        currentLevel.addLast(currentNode.value)
      else:
        currentLevel.addFirst(currentNode.value)

      if currentNode.left: queue.enqueue(currentNode.left)
      if currentNode.right: queue.enqueue(currentNode.right)
    
    result.add(list(currentLevel))
    leftToRight = not leftToRight
    
  return result`
      },
      {
        title: 'Minimum Depth of a Binary Tree',
        pseudoCode: `function findMinimumDepth(root):
  if root is null: return 0
  
  queue = Queue()
  queue.enqueue(root)
  minDepth = 0
  while queue is not empty:
    minDepth += 1
    levelSize = queue.size()
    for i from 0 to levelSize - 1:
      currentNode = queue.dequeue()

      if currentNode.left is null and currentNode.right is null:
        return minDepth
      
      if currentNode.left: queue.enqueue(currentNode.left)
      if currentNode.right: queue.enqueue(currentNode.right)
  
  return minDepth`
      }
    ]
  },
  {
    id: 6,
    name: 'Merge Intervals',
    whenToUse: 'Problems involving overlapping intervals. When you need to merge, insert, or check for conflicts between intervals.',
    approach: 'Sort the intervals by their start time. Iterate through the sorted intervals and merge/compare the current interval with the previous one.',
    example: 'Merge all overlapping intervals in a list.',
    problems: [
      {
        title: 'Merge Overlapping Intervals',
        pseudoCode: `function merge(intervals):
  if intervals.length < 2:
    return intervals
  
  sort intervals by start time
  
  mergedIntervals = []
  start = intervals[0].start
  end = intervals[0].end

  for i from 1 to intervals.length - 1:
    interval = intervals[i]
    if interval.start <= end: // overlapping
      end = max(end, interval.end)
    else: // non-overlapping
      mergedIntervals.add(new Interval(start, end))
      start = interval.start
      end = interval.end
  
  mergedIntervals.add(new Interval(start, end)) // add the last one
  return mergedIntervals`
      },
      {
        title: 'Insert Interval',
        pseudoCode: `function insert(intervals, newInterval):
  merged = []
  i = 0
  
  // add all intervals ending before newInterval starts
  while i < intervals.length and intervals[i].end < newInterval.start:
    merged.add(intervals[i])
    i++
    
  // merge all overlapping intervals
  while i < intervals.length and intervals[i].start <= newInterval.end:
    newInterval.start = min(intervals[i].start, newInterval.start)
    newInterval.end = max(intervals[i].end, newInterval.end)
    i++
  merged.add(newInterval)
  
  // add remaining intervals
  while i < intervals.length:
    merged.add(intervals[i])
    i++
    
  return merged`
      },
      {
        title: 'Intervals Intersection',
        pseudoCode: `function intervalsIntersection(listA, listB):
  intersections = []
  i = 0, j = 0
  while i < listA.length and j < listB.length:
    // check for overlap
    start = max(listA[i].start, listB[j].start)
    end = min(listA[i].end, listB[j].end)
    if start <= end:
      intersections.add(new Interval(start, end))
    
    // move the pointer that has the smaller end time
    if listA[i].end < listB[j].end:
      i++
    else:
      j++
  
  return intersections`
      }
    ]
  },
  {
    id: 7,
    name: 'Cyclic Sort',
    whenToUse: 'Problems involving an array containing numbers in a specific range (e.g., 1 to n). Useful for finding missing, duplicate, or corrupt numbers.',
    approach: 'Iterate through the array. For each element, if it is not at its correct index, swap it with the element at its correct index. Repeat until all elements are correctly placed.',
    example: 'Sort an array of numbers from 1 to n.',
    problems: [
      {
        title: 'Cyclic Sort',
        pseudoCode: `function cyclicSort(nums):
  i = 0
  while i < nums.length:
    correctIndex = nums[i] - 1
    if nums[i] != nums[correctIndex]:
      swap(nums, i, correctIndex)
    else:
      i++`
      },
      {
        title: 'Find the Missing Number (Range 0 to n)',
        pseudoCode: `function findMissingNumber(nums):
  i = 0
  n = nums.length
  while i < n:
    correctIndex = nums[i]
    if nums[i] < n and nums[i] != nums[correctIndex]:
      swap(nums, i, correctIndex)
    else:
      i++
  
  for i from 0 to n - 1:
    if nums[i] != i:
      return i
  
  return n`
      },
      {
        title: 'Find the Duplicate Number',
        pseudoCode: `function findDuplicate(nums):
  i = 0
  while i < nums.length:
    if nums[i] != i + 1:
      correctIndex = nums[i] - 1
      if nums[i] != nums[correctIndex]:
        swap(nums, i, correctIndex)
      else: // found the duplicate
        return nums[i]
    else:
      i++
  return -1`
      }
    ]
  },
   {
    id: 8,
    name: 'In-place Reversal (LinkedList)',
    whenToUse: 'Problems that require reversing a linked list in-place, without using extra memory.',
    approach: 'Iterate through the list, and at each node, reverse its `next` pointer to point to the `previous` node. Keep track of `previous`, `current`, and `next` nodes.',
    example: 'Reverse a singly linked list.',
    problems: [
      {
        title: 'Reverse a Singly Linked List',
        pseudoCode: `function reverse(head):
  previous = null
  current = head
  while current is not null:
    next = current.next // temporarily store the next node
    current.next = previous // reverse the current node's pointer
    previous = current // move pointers one position ahead
    current = next
  return previous // new head`
      },
      {
        title: 'Reverse a Sub-list',
        pseudoCode: `function reverseSublist(head, p, q):
  if p == q: return head

  current = head
  previous = null
  for i from 0 to p - 1:
    previous = current
    current = current.next

  lastNodeOfFirstPart = previous
  lastNodeOfSubList = current
  
  // reverse sublist from p to q
  previous = null
  for i from 0 to q - p + 1:
    next = current.next
    current.next = previous
    previous = current
    current = next
  
  // connect with the first part
  if lastNodeOfFirstPart is not null:
    lastNodeOfFirstPart.next = previous
  else:
    head = previous
    
  // connect with the last part
  lastNodeOfSubList.next = current
  
  return head`
      },
      {
        title: 'Reverse every K-element Sub-list',
        pseudoCode: `function reverseEveryK(head, k):
  if k <= 1 or head is null: return head

  current = head
  previous = null
  while true:
    lastNodeOfPreviousPart = previous
    lastNodeOfSubList = current
    
    // reverse k nodes
    next = null
    i = 0
    while current is not null and i < k:
      next = current.next
      current.next = previous
      previous = current
      current = next
      i++
    
    if lastNodeOfPreviousPart is not null:
      lastNodeOfPreviousPart.next = previous
    else:
      head = previous
      
    lastNodeOfSubList.next = current
    
    if current is null: break
    previous = lastNodeOfSubList
    
  return head`
      }
    ]
  },
  {
    id: 9,
    name: 'Top K Elements',
    whenToUse: 'Finding the top/smallest/most frequent ‘K’ elements in a set.',
    approach: 'Use a Min-Heap or Max-Heap to efficiently keep track of the ‘K’ elements. For top ‘K’ largest, use a Min-Heap. For top ‘K’ smallest, use a Max-Heap.',
    example: 'Find the top ‘K’ largest numbers in an unsorted array.',
    problems: [
      {
        title: 'Top ‘K’ Numbers',
        pseudoCode: `function findKthLargest(nums, k):
  minHeap = MinHeap()
  for num in nums:
    minHeap.add(num)
    if minHeap.size() > k:
      minHeap.poll()
      
  return minHeap.peek()`
      },
      {
        title: '‘K’ Closest Points to the Origin',
        pseudoCode: `function kClosest(points, k):
  maxHeap = MaxHeap based on distance from origin
  
  for point in points:
    maxHeap.add(point)
    if maxHeap.size() > k:
      maxHeap.poll()
      
  return all elements from maxHeap`
      },
      {
        title: 'Top ‘K’ Frequent Numbers',
        pseudoCode: `function topKFrequent(nums, k):
  // build frequency map
  numFrequencyMap = {}
  for num in nums:
    numFrequencyMap[num] = numFrequencyMap.get(num, 0) + 1
    
  minHeap = MinHeap based on frequency
  
  for num, frequency in numFrequencyMap.items():
    minHeap.add([num, frequency])
    if minHeap.size() > k:
      minHeap.poll()
      
  topNumbers = []
  while minHeap is not empty:
    topNumbers.add(minHeap.poll()[0])
    
  return topNumbers`
      }
    ]
  },
  {
    id: 10,
    name: 'Two Heaps',
    whenToUse: 'To find the median or other order statistics in a stream of numbers. Divide a set into two parts and maintain a property about the elements in those parts.',
    approach: 'Use a Max-Heap for the smaller half and a Min-Heap for the larger half. Keep the heaps balanced in size (or differ by at most one). The median can be calculated from the top elements of the heaps.',
    example: 'Find the median of a stream of numbers.',
    problems: [
      {
        title: 'Find the Median of a Number Stream',
        pseudoCode: `class MedianFinder:
  maxHeap = MaxHeap() // for the first half
  minHeap = MinHeap() // for the second half

  function insertNum(num):
    if maxHeap is empty or num <= maxHeap.peek():
      maxHeap.add(num)
    else:
      minHeap.add(num)
      
    // balance heaps
    if maxHeap.size() > minHeap.size() + 1:
      minHeap.add(maxHeap.poll())
    elif minHeap.size() > maxHeap.size():
      maxHeap.add(minHeap.poll())
      
  function findMedian():
    if maxHeap.size() == minHeap.size():
      return (maxHeap.peek() + minHeap.peek()) / 2
    else:
      return maxHeap.peek()`
      },
      {
        title: 'Sliding Window Median',
        pseudoCode: `// Similar to MedianFinder, but with an added remove() function
// and logic to slide the window.
function findSlidingWindowMedian(nums, k):
  medians = []
  finder = MedianFinder()
  windowStart = 0

  for windowEnd from 0 to nums.length - 1:
    finder.insertNum(nums[windowEnd])

    if windowEnd - windowStart + 1 == k:
      medians.add(finder.findMedian())
      elementToRemove = nums[windowStart]
      finder.removeNum(elementToRemove)
      windowStart++
      
  return medians`
      },
      {
        title: 'Maximize Capital',
        pseudoCode: `function findMaximizedCapital(capitals, profits, initialCapital, k):
  minCapitalHeap = MinHeap of (capital, profit) pairs
  maxProfitHeap = MaxHeap of profits
  
  // insert all projects into minCapitalHeap
  for i from 0 to capitals.length - 1:
    minCapitalHeap.add((capitals[i], profits[i]))
    
  availableCapital = initialCapital
  for i from 0 to k - 1:
    // move all affordable projects to maxProfitHeap
    while minCapitalHeap is not empty and minCapitalHeap.peek().capital <= availableCapital:
      maxProfitHeap.add(minCapitalHeap.poll().profit)
      
    if maxProfitHeap is empty:
      break
      
    availableCapital += maxProfitHeap.poll()
    
  return availableCapital`
      }
    ]
  },
  {
    id: 11,
    name: 'Modified Binary Search',
    whenToUse: 'Searching in a sorted array (or a data structure that behaves like one) with a twist, such as being rotated, having duplicates, or searching for a specific boundary (first/last occurrence).',
    approach: 'The core is standard binary search (low, high, mid). The modification comes in how you adjust the `low` and `high` pointers based on the problem\'s specific conditions, which often involves comparing `arr[mid]` with `arr[low]` or `arr[high]`.',
    example: 'Find an element in a rotated sorted array.',
    problems: [
      {
        title: 'Order-agnostic Binary Search',
        pseudoCode: `function binarySearch(arr, key):
  start = 0, end = arr.length - 1
  isAscending = arr[start] < arr[end]
  
  while start <= end:
    mid = start + floor((end - start) / 2)
    if key == arr[mid]: return mid
    
    if isAscending:
      if key < arr[mid]: end = mid - 1
      else: start = mid + 1
    else: // descending
      if key > arr[mid]: end = mid - 1
      else: start = mid + 1
      
  return -1`
      },
      {
        title: 'Ceiling of a Number',
        pseudoCode: `function searchCeiling(arr, key):
  if key > arr[arr.length - 1]: return -1
  
  start = 0, end = arr.length - 1
  while start <= end:
    mid = start + floor((end - start) / 2)
    if key < arr[mid]: end = mid - 1
    else if key > arr[mid]: start = mid + 1
    else: return mid
    
  return start`
      },
      {
        title: 'Search in a Rotated Sorted Array',
        pseudoCode: `function searchRotated(arr, key):
  start = 0, end = arr.length - 1
  while start <= end:
    mid = start + floor((end - start) / 2)
    if arr[mid] == key: return mid
    
    if arr[start] <= arr[mid]: // left side is sorted
      if key >= arr[start] and key < arr[mid]:
        end = mid - 1
      else:
        start = mid + 1
    else: // right side is sorted
      if key > arr[mid] and key <= arr[end]:
        start = mid + 1
      else:
        end = mid - 1
        
  return -1`
      }
    ]
  },
  {
    id: 12,
    name: 'Topological Sort',
    whenToUse: 'Problems involving dependencies between items, where you need to find a linear ordering. Examples include task scheduling, course prerequisites, or build systems.',
    approach: 'Use Breadth-First Search (BFS). First, build a graph and find all sources (nodes with 0 in-degrees). Add sources to a queue. Process the queue: for each source, add it to the sorted list and decrement the in-degree of its neighbors. If a neighbor\'s in-degree becomes 0, add it to the queue.',
    example: 'Find the order of tasks to complete given dependencies.',
    problems: [
      {
        title: 'Topological Sort (Task Scheduling)',
        pseudoCode: `function topologicalSort(vertices, edges):
  sortedOrder = []
  if vertices <= 0: return sortedOrder
  
  // Initialize graph and in-degrees
  inDegree = map of {vertex: 0}
  graph = map of {vertex: []}
  for v in vertices:
    inDegree[v] = 0
    graph[v] = []
  
  // Build graph and populate in-degrees
  for edge in edges:
    parent, child = edge[0], edge[1]
    graph[parent].add(child)
    inDegree[child] += 1
  
  // Find all sources (in-degree 0)
  sources = Queue()
  for vertex, degree in inDegree.items():
    if degree == 0:
      sources.enqueue(vertex)
      
  // Process sources
  while sources is not empty:
    vertex = sources.dequeue()
    sortedOrder.add(vertex)
    for child in graph[vertex]:
      inDegree[child] -= 1
      if inDegree[child] == 0:
        sources.enqueue(child)
        
  // Check for cycle
  if len(sortedOrder) != vertices:
    return [] // cycle detected
    
  return sortedOrder`
      },
      {
        title: 'Course Schedule (Can Finish?)',
        pseudoCode: `// Use the same topological sort logic.
// If the length of the sortedOrder list is equal to the number of courses,
// it means we can finish all courses. Otherwise, there's a cycle and it's impossible.
function canFinish(numCourses, prerequisites):
  sortedOrder = topologicalSort(numCourses, prerequisites)
  return len(sortedOrder) == numCourses`
      },
      {
        title: 'Alien Dictionary',
        pseudoCode: `// 1. Build graph and in-degrees from the list of words.
//    Compare adjacent words to find ordering rules. e.g., "wrt", "wrf" => t -> f
// 2. Perform topological sort on the graph of characters.
// 3. Return the sorted list of characters as a string.
//    If a cycle is detected, return an empty string.

function alienOrder(words):
  // ... initialization of inDegree and graph for all unique chars ...

  // Build graph by comparing adjacent words
  for i from 0 to words.length - 2:
    word1 = words[i]
    word2 = words[i+1]
    for j from 0 to min(len(word1), len(word2)):
      parent, child = word1[j], word2[j]
      if parent != child:
        graph[parent].add(child)
        inDegree[child] += 1
        break // only first difference matters

  // ... perform topological sort as in the main example ...
  
  // return result`
      }
    ]
  },
  {
    id: 13,
    name: 'Subsets / Combinations',
    whenToUse: 'Problems asking for all permutations, combinations, or subsets of a given set of elements.',
    approach: 'Typically solved with recursion or backtracking. For subsets, a common approach is to start with an empty set, then iterate through the input elements, adding each element to all existing subsets to create new ones.',
    example: 'Find all subsets of a given set.',
    problems: [
      {
        title: 'Find all Subsets',
        pseudoCode: `function findSubsets(nums):
  subsets = [[]] // start with the empty set
  for currentNumber in nums:
    // we will take all existing subsets and insert the current number in them
    // to create new subsets
    n = len(subsets)
    for i from 0 to n-1:
      set = list(subsets[i])
      set.add(currentNumber)
      subsets.add(set)
  return subsets`
      },
      {
        title: 'Subsets With Duplicates',
        pseudoCode: `function findSubsetsWithDuplicates(nums):
  sort(nums)
  subsets = [[]]
  startIndex = 0
  endIndex = 0
  for i from 0 to len(nums)-1:
    startIndex = 0
    // if current number is same as previous, create subsets only from the subsets
    // added in the previous step
    if i > 0 and nums[i] == nums[i-1]:
      startIndex = endIndex + 1
    endIndex = len(subsets) - 1
    for j from startIndex to endIndex:
      set = list(subsets[j])
      set.add(nums[i])
      subsets.add(set)
  return subsets`
      },
      {
        title: 'Generate Parentheses',
        pseudoCode: `function generateParentheses(n):
  result = []
  generate(n, 0, 0, "", result)
  return result

function generate(n, openCount, closeCount, currentString, result):
  if len(currentString) == 2 * n:
    result.add(currentString)
    return

  if openCount < n:
    generate(n, openCount + 1, closeCount, currentString + '(', result)
  
  if closeCount < openCount:
    generate(n, openCount, closeCount + 1, currentString + ')', result)`
      }
    ]
  },
    {
    id: 14,
    name: 'Bitwise XOR',
    whenToUse: 'Problems involving finding a missing number, a single unique number in a list of duplicates, or dealing with properties of binary representations of numbers.',
    approach: 'XOR has useful properties: `x ^ x = 0` and `x ^ 0 = x`. By XORing all numbers in a set, you can cancel out duplicates and isolate unique elements.',
    example: 'Find the single number in an array where every other number appears twice.',
    problems: [
      {
        title: 'Single Number',
        pseudoCode: `function findSingleNumber(arr):
  num = 0
  for i in arr:
    num = num ^ i
  return num`
      },
      {
        title: 'Two Single Numbers',
        pseudoCode: `function findTwoSingleNumbers(nums):
  // 1. XOR all numbers to get the XOR of the two single numbers
  n1xn2 = 0
  for num in nums:
    n1xn2 ^= num
    
  // 2. Find a bit that is set (a '1') in n1xn2.
  //    This bit must be different between the two single numbers.
  rightmostSetBit = 1
  while (rightmostSetBit & n1xn2) == 0:
    rightmostSetBit = rightmostSetBit << 1
    
  // 3. Partition numbers into two groups and XOR each group.
  num1 = 0, num2 = 0
  for num in nums:
    if (num & rightmostSetBit) != 0: // the bit is set
      num1 ^= num
    else: // the bit is not set
      num2 ^= num
      
  return [num1, num2]`
      },
      {
        title: 'Complement of Base 10 Number',
        pseudoCode: `function bitwiseComplement(num):
  if num == 0: return 1
  
  bitCount = 0
  n = num
  while n > 0:
    bitCount++
    n = n >> 1
    
  // all_bits_set is a number with 'bitCount' 1s.
  // e.g., if num is 8 (1000), bitCount is 4, all_bits_set is 15 (1111)
  all_bits_set = 2^bitCount - 1
  
  // complement is all_bits_set XOR num
  return all_bits_set ^ num`
      }
    ]
  },
  {
    id: 15,
    name: 'Dynamic Programming - 0/1 Knapsack',
    whenToUse: 'Problems where you need to make a selection of items to maximize or minimize a value, given a constraint (like weight or capacity). For each item, you have two choices: include it or not.',
    approach: 'Typically solved using a 2D DP table, where `dp[i][c]` represents the maximum value using the first `i` items with a capacity of `c`. The state transition is `dp[i][c] = max(dp[i-1][c], value[i] + dp[i-1][c - weight[i]])`.',
    example: 'Given weights and values of n items, put these items in a knapsack of capacity W to get the maximum total value.',
    problems: [
      {
        title: '0/1 Knapsack',
        pseudoCode: `function solveKnapsack(profits, weights, capacity):
  n = len(profits)
  dp = 2D array of size (n x capacity+1) initialized to 0
  
  // populate for capacity 0 (can't take any items) - already 0
  
  // process first item
  for c from 1 to capacity:
    if weights[0] <= c:
      dp[0][c] = profits[0]
      
  // process remaining items
  for i from 1 to n-1:
    for c from 1 to capacity:
      profit1 = 0 // profit if we include item i
      if weights[i] <= c:
        profit1 = profits[i] + dp[i-1][c-weights[i]]
      
      profit2 = dp[i-1][c] // profit if we exclude item i
      
      dp[i][c] = max(profit1, profit2)
      
  return dp[n-1][capacity]`
      },
      {
        title: 'Equal Subset Sum Partition',
        pseudoCode: `function canPartition(nums):
  sum = sum(nums)
  if sum is odd: return false
  targetSum = sum / 2
  
  // This is now a knapsack problem: find a subset with sum equal to targetSum
  dp = 1D array of size (targetSum+1) initialized to false, dp[0] = true
  
  for num in nums:
    for s from targetSum down to num:
      dp[s] = dp[s] or dp[s-num]
      
  return dp[targetSum]`
      },
      {
        title: 'Count of Subset Sum',
        pseudoCode: `function countSubsets(nums, sum):
  n = len(nums)
  dp = 2D array of size (n x sum+1) initialized to 0
  
  // populate for sum 0 (empty set)
  for i from 0 to n-1:
    dp[i][0] = 1
    
  // process first item
  for s from 1 to sum:
    if nums[0] == s: dp[0][s] = 1
  
  // process remaining items
  for i from 1 to n-1:
    for s from 1 to sum:
      dp[i][s] = dp[i-1][s] // exclude the number
      if s >= nums[i]:
        dp[i][s] += dp[i-1][s - nums[i]] // include the number
        
  return dp[n-1][sum]`
      }
    ]
  },
  // Add 5 more patterns to reach 20
  {
    id: 16,
    name: 'HashMap/HashSet Patterns',
    whenToUse: 'When you need efficient lookups, insertions, or deletions (O(1) on average). Useful for counting frequencies, finding duplicates, or checking for the existence of elements.',
    approach: 'Use a HashMap (dictionary/map) to store key-value pairs or a HashSet to store unique elements. Iterate through the data, using the hash-based structure to store and retrieve information quickly.',
    example: 'Find the first non-repeating character in a string.',
    problems: [
      {
        title: 'Two Sum',
        pseudoCode: `function twoSum(nums, target):
  map = HashMap() // value -> index
  for i from 0 to len(nums)-1:
    complement = target - nums[i]
    if complement in map:
      return [map[complement], i]
    map[nums[i]] = i
  return []`
      },
      {
        title: 'Isomorphic Strings',
        pseudoCode: `function isIsomorphic(s, t):
  if len(s) != len(t): return false
  
  mapS_T = HashMap()
  mapT_S = HashMap()
  
  for i from 0 to len(s)-1:
    charS = s[i]
    charT = t[i]
    
    if charS in mapS_T and mapS_T[charS] != charT:
      return false
    if charT in mapT_S and mapT_S[charT] != charS:
      return false
      
    mapS_T[charS] = charT
    mapT_S[charT] = charS
    
  return true`
      },
      {
        title: 'Group Anagrams',
        pseudoCode: `function groupAnagrams(strs):
  anagramMap = HashMap() // sorted_string -> list_of_anagrams
  
  for s in strs:
    sortedS = sort_string(s)
    if sortedS not in anagramMap:
      anagramMap[sortedS] = []
    anagramMap[sortedS].add(s)
    
  return list of values from anagramMap`
      }
    ]
  },
  {
    id: 17,
    name: 'Matrix Traversal (Islands)',
    whenToUse: 'Problems involving a 2D grid or matrix where you need to find connected components, paths, or areas. Often framed as finding "islands" or "continents".',
    approach: 'Iterate through each cell of the matrix. If you find a starting point of a component (e.g., a \'1\' for an island), start a traversal (DFS or BFS) from that cell. Mark all visited cells within that component to avoid recounting them.',
    example: 'Count the number of islands in a grid.',
    problems: [
      {
        title: 'Number of Islands',
        pseudoCode: `function numIslands(grid):
  if not grid: return 0
  rows, cols = len(grid), len(grid[0])
  count = 0
  
  for r from 0 to rows-1:
    for c from 0 to cols-1:
      if grid[r][c] == '1':
        count += 1
        dfs(grid, r, c)
        
  return count

function dfs(grid, r, c):
  if r<0 or c<0 or r>=len(grid) or c>=len(grid[0]) or grid[r][c]=='0':
    return
  grid[r][c] = '0' // Sink the island
  dfs(grid, r+1, c)
  dfs(grid, r-1, c)
  dfs(grid, r, c+1)
  dfs(grid, r, c-1)`
      },
      {
        title: 'Max Area of Island',
        pseudoCode: `function maxAreaOfIsland(grid):
  maxArea = 0
  rows, cols = len(grid), len(grid[0])
  
  for r from 0 to rows-1:
    for c from 0 to cols-1:
      if grid[r][c] == 1:
        maxArea = max(maxArea, dfsArea(grid, r, c))
        
  return maxArea

function dfsArea(grid, r, c):
  if r<0 or c<0 or r>=len(grid) or c>=len(grid[0]) or grid[r][c]==0:
    return 0
  grid[r][c] = 0 // Mark as visited
  return (1 + dfsArea(grid, r+1, c) + dfsArea(grid, r-1, c) + 
              dfsArea(grid, r, c+1) + dfsArea(grid, r, c-1))`
      },
      {
        title: 'Rotting Oranges',
        pseudoCode: `function orangesRotting(grid):
  queue = Queue()
  fresh_oranges = 0
  rows, cols = len(grid), len(grid[0])
  
  for r from 0 to rows-1:
    for c from 0 to cols-1:
      if grid[r][c] == 2: queue.enqueue((r, c))
      if grid[r][c] == 1: fresh_oranges += 1
  
  if fresh_oranges == 0: return 0
  minutes = -1
  
  while queue is not empty:
    level_size = len(queue)
    for _ in range(level_size):
      r, c = queue.dequeue()
      for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
        nr, nc = r+dr, c+dc
        if 0<=nr<rows and 0<=nc<cols and grid[nr][nc]==1:
          grid[nr][nc] = 2
          fresh_oranges -= 1
          queue.enqueue((nr, nc))
    minutes += 1
    
  return minutes if fresh_oranges == 0 else -1`
      }
    ]
  },
  {
    id: 18,
    name: 'K-way Merge',
    whenToUse: 'Merging multiple sorted lists or arrays into a single sorted list. Finding the smallest element across several sorted collections.',
    approach: 'Use a Min-Heap. Insert the first element from each of the ‘K’ lists into the heap. Then, repeatedly extract the minimum element from the heap, add it to the result list, and insert the next element from the same list that the extracted element came from.',
    example: 'Merge ‘K’ sorted linked lists.',
    problems: [
      {
        title: 'Merge K Sorted Lists',
        pseudoCode: `function mergeKLists(lists):
  minHeap = MinHeap()
  
  // Put the root of each list in the min heap
  for root in lists:
    if root is not null:
      minHeap.add(root)
      
  resultHead, resultTail = null, null
  while minHeap is not empty:
    node = minHeap.poll()
    if resultHead is null:
      resultHead = resultTail = node
    else:
      resultTail.next = node
      resultTail = resultTail.next
      
    if node.next is not null:
      minHeap.add(node.next)
      
  return resultHead`
      },
      {
        title: 'Kth Smallest Number in M Sorted Lists',
        pseudoCode: `function findKthSmallest(lists, k):
  minHeap = MinHeap of (number, listIndex, elementIndex)
  
  // push the first element of each list
  for i from 0 to len(lists)-1:
    if lists[i] is not empty:
      minHeap.add((lists[i][0], i, 0))
      
  numberCount = 0
  result = -1
  while minHeap is not empty:
    number, listIndex, elementIndex = minHeap.poll()
    numberCount += 1
    if numberCount == k:
      result = number
      break
      
    if elementIndex+1 < len(lists[listIndex]):
      nextElement = lists[listIndex][elementIndex+1]
      minHeap.add((nextElement, listIndex, elementIndex+1))
      
  return result`
      },
      {
        title: 'Smallest Number Range',
        pseudoCode: `function smallestRange(lists):
  minHeap = MinHeap of (number, listIndex, elementIndex)
  currentMax = -infinity
  
  for i from 0 to len(lists)-1:
    minHeap.add((lists[i][0], i, 0))
    currentMax = max(currentMax, lists[i][0])
    
  rangeStart, rangeEnd = 0, infinity
  
  while len(minHeap) == len(lists):
    num, listIdx, elemIdx = minHeap.poll()
    
    if rangeEnd - rangeStart > currentMax - num:
      rangeStart = num
      rangeEnd = currentMax
      
    if elemIdx + 1 < len(lists[listIdx]):
      nextNum = lists[listIdx][elemIdx + 1]
      minHeap.add((nextNum, listIdx, elemIdx + 1))
      currentMax = max(currentMax, nextNum)
      
  return [rangeStart, rangeEnd]`
      }
    ]
  },
  {
    id: 19,
    name: 'Dynamic Programming - Counting',
    whenToUse: 'Problems that ask "how many ways" there are to do something. These problems often have overlapping subproblems that can be solved and stored to avoid re-computation.',
    approach: 'Define a DP state, often `dp[i]`, which represents the number of ways to solve the problem for size `i`. Find a recurrence relation that expresses `dp[i]` in terms of smaller subproblems (e.g., `dp[i-1]`, `dp[i-2]`).',
    example: 'Climbing stairs: How many distinct ways can you climb to the top if you can take 1 or 2 steps at a time?',
    problems: [
      {
        title: 'Climbing Stairs',
        pseudoCode: `function climbStairs(n):
  if n <= 2: return n
  
  dp = array of size n+1
  dp[1] = 1
  dp[2] = 2
  
  for i from 3 to n:
    dp[i] = dp[i-1] + dp[i-2]
    
  return dp[n]`
      },
      {
        title: 'Coin Change 2 (Number of Combinations)',
        pseudoCode: `function change(amount, coins):
  dp = array of size amount+1, initialized to 0
  dp[0] = 1 // one way to make amount 0 (with no coins)
  
  for coin in coins:
    for i from coin to amount:
      dp[i] += dp[i - coin]
      
  return dp[amount]`
      },
      {
        title: 'Unique Paths',
        pseudoCode: `function uniquePaths(m, n):
  dp = 2D grid of size m x n
  
  // Initialize first row and first column to 1
  for i from 0 to m-1: dp[i][0] = 1
  for j from 0 to n-1: dp[0][j] = 1
  
  for i from 1 to m-1:
    for j from 1 to n-1:
      dp[i][j] = dp[i-1][j] + dp[i][j-1]
      
  return dp[m-1][n-1]`
      }
    ]
  },
  {
    id: 20,
    name: 'Longest Common Substring/Subsequence',
    whenToUse: 'Problems that involve comparing two sequences (strings, arrays) to find the longest shared part, either contiguous (substring) or not (subsequence).',
    approach: 'Use a 2D DP table. For Substring, `dp[i][j]` is the length of the common substring ending at `s1[i-1]` and `s2[j-1]`. For Subsequence, `dp[i][j]` is the length of the common subsequence for `s1[0..i-1]` and `s2[0..j-1]`.',
    example: 'Find the length of the longest common subsequence between two strings.',
    problems: [
      {
        title: 'Longest Common Subsequence (LCS)',
        pseudoCode: `function longestCommonSubsequence(text1, text2):
  n1, n2 = len(text1), len(text2)
  dp = 2D grid of size (n1+1) x (n2+1) initialized to 0
  
  for i from 1 to n1:
    for j from 1 to n2:
      if text1[i-1] == text2[j-1]:
        dp[i][j] = 1 + dp[i-1][j-1]
      else:
        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
  return dp[n1][n2]`
      },
      {
        title: 'Longest Common Substring',
        pseudoCode: `function longestCommonSubstring(text1, text2):
  n1, n2 = len(text1), len(text2)
  dp = 2D grid of size (n1+1) x (n2+1) initialized to 0
  maxLength = 0
  
  for i from 1 to n1:
    for j from 1 to n2:
      if text1[i-1] == text2[j-1]:
        dp[i][j] = 1 + dp[i-1][j-1]
        maxLength = max(maxLength, dp[i][j])
      else:
        dp[i][j] = 0 // reset if not contiguous
        
  return maxLength`
      },
      {
        title: 'Edit Distance',
        pseudoCode: `function minDistance(word1, word2):
  n1, n2 = len(word1), len(word2)
  dp = 2D grid of size (n1+1) x (n2+1)
  
  // Base cases
  for i from 0 to n1: dp[i][0] = i
  for j from 0 to n2: dp[0][j] = j
  
  for i from 1 to n1:
    for j from 1 to n2:
      if word1[i-1] == word2[j-1]:
        dp[i][j] = dp[i-1][j-1]
      else:
        dp[i][j] = 1 + min(dp[i-1][j],      // Deletion
                             dp[i][j-1],      // Insertion
                             dp[i-1][j-1])     // Replacement
                             
  return dp[n1][n2]`
      }
    ]
  }
];
