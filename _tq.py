import firebase_admin, os
from firebase_admin import credentials, firestore
os.environ.setdefault("MENSIA_CODE_PEPPER","test-pepper")
if not firebase_admin._apps:
    firebase_admin.initialize_app(credentials.Certificate("firebase_service_account.json"))
db = firestore.client()
import quotas as Q

UID = "usr_TESTQUOTA1"
db.collection(Q.USERS).document(UID).delete()
for kind,expect in [("extra","weekly"),("special","unlimited")]:
    pass

# 1. free lifetime: 5 search allowed, 6th blocked
db.collection(Q.USERS).document(UID).delete()
ok=[]
for i in range(6):
    a,msg,snap = Q.check_and_consume(db,UID,"t@e.com","search")
    ok.append(a)
print("free search 6 tries ->",ok, "| expect [T,T,T,T,T,F]")
print("  msg:",msg)
print("  snap:",Q.snapshot(Q.read_user(db,UID)))

# 2. deep research 1 lifetime
print("dr1:",Q.check_and_consume(db,UID,"t@e.com","deep research")[0], "dr2:",Q.check_and_consume(db,UID,"t@e.com","deep research")[0])
# 3. vin base then investigator (shared pool of 1)
print("vinbase:",Q.check_and_consume(db,UID,"t@e.com","vin","base")[0], "vininv:",Q.check_and_consume(db,UID,"t@e.com","vin","investigator")[0])

# 4. code: generate + store + redeem
code,h = Q.generate_code("extra")
Q.store_code(db,"extra",code,h)
db.collection(Q.USERS).document(UID).delete()
ok,msg,snap = Q.redeem_code(db,UID,"t@e.com",code)
print("redeem:",ok,msg,"plan:",snap["plan"],"window:",snap["window"])
# consume to weekly cap 3 vin_base
res=[Q.check_and_consume(db,UID,"t@e.com","vin","base")[0] for _ in range(4)]
print("extra vin_base x4 ->",res,"| expect [T,T,T,F]")
r2=[Q.check_and_consume(db,UID,"t@e.com","vin","investigator")[0] for _ in range(8)]
print("extra vin_inv x8 ->",r2,"| expect 7 T then F")
snap=Q.snapshot(Q.read_user(db,UID)); print("  rows:",snap["rows"])

# 5. single use: second account blocked
UID2="usr_TESTQUOTA2"
db.collection(Q.USERS).document(UID2).delete()
ok2,msg2,_=Q.redeem_code(db,UID2,"o@e.com",code)
print("reuse by 2nd account:",ok2,msg2)

# 6. special code -> unlimited, no leak of word 'special'
scode,sh=Q.generate_code("special"); Q.store_code(db,"special",scode,sh)
db.collection(Q.USERS).document(UID2).delete()
ok3,msg3,snap3=Q.redeem_code(db,UID2,"o@e.com",scode)
print("special redeem:",ok3,"plan:",snap3["plan"],"window:",snap3["window"])
print("  50 search calls allowed:",all(Q.check_and_consume(db,UID2,"o@e.com","search")[0] for _ in range(50)))

# cleanup
for u in (UID,UID2):
    d=db.collection(Q.USERS).document(u); d.exists and d.delete()
for x in (h,sh): db.collection(Q.CODES).document(x).delete()
print("cleaned up")
