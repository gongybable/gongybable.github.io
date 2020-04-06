## Tables
Use SQL, for transaction support and relations in the data.

1. **Order**: OrderID (PK), StoreID, UserID, ShopperID, CreationDate, Status (new, pending, rejected, approved)

2. **Store**: StoreID (PK), Name, Address1 (varchar 255) Not Null, Address2, City, State, Country, PostalCode

3. **Item**: ItemID (PK), StoreID, UnitPrice (decimal 2), BarCode, Name, Category

4. **OrderItem**: OrderID (PK), ItemID (PK), OrderUnitPrice, Quantity


API Gateway -----> payment service ----> SQL DB
                         |                  |
                         |                  |
                         |                  |
                        cache               |
                         |      update  order services
                         |------------- (workers)