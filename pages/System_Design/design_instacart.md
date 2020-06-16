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


