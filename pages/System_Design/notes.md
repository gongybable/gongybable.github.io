# API
1. Methods
    * GET - get
    * POST - create
    * PUT - replace entirely
    * PATCH - partial update
    * DELETE - delete

2. Use plural nouns

3. Use sub-resources for relations

4. Use HATEOAS (have links for resources for better navigation)

5. Use query parameters for filtering, sorting and pagination

# Design
* Estimates:
    * traffic estimate
    * storage estimate
    * bandwidth estimate
    * memory estimate (20, 80 rule)

* Reverse Proxy(web server)
    * cache
    * compression
    * serve static content

* Use a seperate server (and URL) to serve static images; to allow easier migration to CDN later on.
    - CDN Pull - when there is lots of traffic
    - CDN Push - when there is less traffic and content does not update too often

* Split read/write apis to scale them seperately

* Use map reduce / multi thread to parallel process with big data

* DB Design
    * float vs decimal: float is approximate value, decimal is exact numerical value
    * Master - write; Slave - read
    * Use caches for reads
    * Federation - split DBs by function
    * Sharding - split users table by last names or by location
    * Denormalization - imporve read performance at the expense of writes by allowing data redundency
    * Store 3 month of data in DB and older data in a data warehouse.

* Logging Design
    1. Each action (i.e. tinyUrl access request) triggers an “Event”
    2. Each “Event” is sent to a messaging queue system (e.g. Apache Kafka)
        * Events are ETLs into a cold storage (e.g. S3/HDFS) and processed offline (MapReduce/Spark) to generate ALL the metrics you want
        * Events are processed in near realtime (e.g. Apache Storm) for quick analysis
    4. Visualization/Reporting/Notifications

# LeetCode
```python
arr.sort(key=lambda x: -x[1])
c.isdigit()

import random
random.randint(lo, hi)


def morris_traversal(root):
    # inorder traversal
    curr = root
    while curr:
        if curr.left:
            pred = curr.left
            while pred.right and pred.right != curr:
                pred = pred.right
            if not pred.right:
                pred.right = curr
                curr = curr.left
            else:
                visit(curr)
                pred.right = None
                curr = curr.right
        else:
            visit(curr)
            curr = curr.right
```