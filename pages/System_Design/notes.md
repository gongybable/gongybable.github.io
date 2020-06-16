## API
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

## Design
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
        * Events are ETLs into a cloud storage (e.g. S3/HDFS) and processed offline (MapReduce/Spark) to generate ALL the metrics you want
        * Events are processed in near realtime (e.g. Apache Storm) for quick analysis
    4. Visualization/Reporting/Notifications

## HTTP vs WebSocket

Slow page load speeds: Every Web browser has a limited “resource” pool for all HTTP-based requests including HTML, CSS, fonts, javascript libraries, images and other media as well as XHR and JSONP. This pool is organized by resource priority where the priority of XHR or JSONP requests is the least, and the pool has limits on the number of parallel network requests. This means that excessive usage of third-party HTTP-based libraries including analytics libraries may slow down your website significantly because these libraries could flood the pool with a lot of requests. On the other hand, low priority HTTP requests may wait in the pool for long time (up to a few minutes), which results in losing user sessions and events for those who browse the website for only a short time eg. readers who read only 1 article in a session.

Increased bandwidth / data consumption: HTTP communication has huge data transfer overhead sending protocol-related information like headers with every request/response.

## Pull vs Push
1. Pull: do not know when to pull (long-polling); stores may have different technologies and infrastructure, difficult to manage the interfaces connecting to them; what if store changes their tech stack and infrastructure?

2. Push: push when there is a change; one interface for all stores; rely on store's services, need to monitor their service availability; Push every 5 mins?

## Kafka
1. stores data on disk
2. kafka borkers distributed across multiple machines
3. set kafka to only accept writes if the dada is commited to all the machines
4. only commit the kafka offset (where the data is read from) after the client finished the work
5. one kafka partition per worker

## Distributed System

### API Gateway / Backend for Frontend
One API gateway for Android, one for IOS, one for web clients and one for 3rd party clients.

1. Authentication - JWT authentication
2. SSL termination, allow different internal protocals
3. Load balancing
4. Abstraction
5. Insulation - changes in backend services does not require client code change
6. Caching
7. A/B testing
8. API versioning
9. Rate limit and analytics for billing
10. Circuit breaking

### Service Registry
1. Each container sends a heartbeat to the registry service with its status and IP address information
2. The information is sent every 5 mins
3. In the registry, if an entry is not updated longer than 5 mins, we know the container is dead


### Inter Service Communication
API GW <===> MS1 <===> MS2 <===> MS3

Sync Call:
* Easy to implement
* Real time
* Long response time
* Service availability: one service down cause error in the response, services denpend on each other.

Async Call using Queues:
* Faster APIs
* Decoupled services
* No need for service discovery
* Complex design
* Process latency
* Monitoring cost

### Circuit Break (Hystrix)
MS1 (CB) <===> MS2 (CB) <===> MS3

* Prevent error propagating backwards
* Return cached / default response for the failed service
* Redirect to another similar service for the failed service
* Gives time for the failed service to recover, and after 5 mins checks the service again

### Service Mesh (sidecar of the container)
 - control plane
 - data plane (where data communication happens)
1. Load balancing
2. service discovery
3. circuit break
4. collect metrics
5. retries
6. timeout
7. proxy
