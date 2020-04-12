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
















