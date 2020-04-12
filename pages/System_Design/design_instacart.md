## Tables
Use SQL, for transaction support and relations in the data.
### Cache
(OrderID, StoreID): [Amout_lo, Amout_hi]

### Table
1. **Order**: OrderID (PK), StoreID, UserID, CreationTime, Status (new, running, pending, rejected, approved, confirmed), QueuedTime

2. **Store**: StoreID (PK), Name, Address1 (varchar 255) Not Null, Address2, City, State, Country, PostalCode

3. **Item**: ItemID (PK), StoreID, UnitPrice (decimal 2), BarCode, Name, Category

4. **ClientOrderItem**: OrderID (PK), ItemID (PK), OrderUnitPrice, Quantity

5. **Payment**: PaymentID, OrderID, StoreID, Amount, Status (rejected, approved)

6. **Shopper**: ShopperID (PK), OrderID (PK), Rating

NoSQL?
7. **ShopperOrderDetails**: OrderID (TransactionID) storeID Datetime Amount

8. **ShopperOrderItem**: OrderID TransactionID Name UnitPrice Quantity

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

