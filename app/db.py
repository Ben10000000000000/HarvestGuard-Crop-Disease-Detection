"""MongoDB persistence layer for HarvestGuard.

A small compatibility adapter keeps the existing route code readable while all
production data is stored in MongoDB. SQLite is used only by the test suite when
an explicit DATABASE path is supplied.
"""
from __future__ import annotations
import re, sqlite3
from datetime import datetime, timezone
from flask import current_app, g

class Result:
    def __init__(self, rows=None, lastrowid=None): self.rows=list(rows or []); self.lastrowid=lastrowid
    def fetchone(self): return self.rows[0] if self.rows else None
    def fetchall(self): return self.rows

class MongoDB:
    def __init__(self, db): self.db=db; self.lastrowid=None
    def commit(self): pass
    def close(self): pass
    def _id(self, table):
        return self.db.counters.find_one_and_update({'_id':table},{'$inc':{'n':1}},upsert=True,return_document=__import__('pymongo').ReturnDocument.AFTER)['n']
    def execute(self, sql, args=()):
        s=' '.join(sql.split()); low=s.lower(); a=list(args)
        if low.startswith('select 1 from users where email'):
            d=self.db.users.find_one({'email':a[0]}, {'_id':0,'id':1}); return Result([{'1':1}] if d else [])
        if low.startswith('select * from users where email'):
            return Result([self.db.users.find_one({'email':a[0]}, {'_id':0})] if self.db.users.find_one({'email':a[0]}) else [])
        if low.startswith('select id from users where email'):
            d=self.db.users.find_one({'email':a[0]}, {'_id':0,'id':1}); return Result([d] if d else [])
        if low.startswith('select id, name, email, created_at from users where id'):
            d=self.db.users.find_one({'id':a[0]}, {'_id':0,'id':1,'name':1,'email':1,'created_at':1}); return Result([d] if d else [])
        if low.startswith('select email, name from users where id'):
            d=self.db.users.find_one({'id':a[0]}, {'_id':0,'email':1,'name':1}); return Result([d] if d else [])
        if low.startswith('select created_at, last_login_at from users where id'):
            d=self.db.users.find_one({'id':a[0]}, {'_id':0,'created_at':1,'last_login_at':1}); return Result([d] if d else [])
        if low.startswith('select * from users where id'):
            d=self.db.users.find_one({'id':a[0]}, {'_id':0}); return Result([d] if d else [])
        if low.startswith('select * from otps where user_id'):
            d=self.db.otps.find_one({'user_id':a[0],'purpose':a[1],'used':0},{'_id':0},sort=[('id',-1)]); return Result([d] if d else [])
        if low.startswith('select * from password_resets where token_hash'):
            d=self.db.password_resets.find_one({'token_hash':a[0],'used':0},{'_id':0}); return Result([d] if d else [])
        if low.startswith('select count(*) n from predictions'):
            filt={'user_id':a[0]}
            if 'and is_healthy = 1' in low: filt['is_healthy']=1
            elif 'and is_healthy = 0' in low: filt['is_healthy']=0
            n=self.db.predictions.count_documents(filt); return Result([{'n':n}])
        if low.startswith('select count(*) n, coalesce(sum(is_healthy)'):
            d=list(self.db.predictions.aggregate([{'$match':{'user_id':a[0]}},{'$group':{'_id':None,'n':{'$sum':1},'healthy':{'$sum':'$is_healthy'},'conf':{'$avg':'$confidence'}}}]))
            return Result([{'n':d[0]['n'],'healthy':d[0]['healthy'],'conf':d[0].get('conf') or 0.0}] if d else [{'n':0,'healthy':0,'conf':0.0}])
        if low.startswith('select condition, crop, count(*) n from predictions'):
            docs=list(self.db.predictions.aggregate([{'$match':{'user_id':a[0],'is_healthy':0}},{'$group':{'_id':'$label','condition':{'$first':'$condition'},'crop':{'$first':'$crop'},'n':{'$sum':1}}},{'$sort':{'n':-1}},{'$limit':5},{'$project':{'_id':0,'condition':1,'crop':1,'n':1}}])); return Result(docs)
        if low.startswith('select * from predictions where user_id = ? order by id desc limit 6'):
            return Result(list(self.db.predictions.find({'user_id':a[0]},{'_id':0}).sort('id',-1).limit(6)))
        if low.startswith('select * from predictions where id = ? and user_id = ?'):
            d=self.db.predictions.find_one({'id':a[0],'user_id':a[1]},{'_id':0}); return Result([d] if d else [])
        if low.startswith('select count(*) n from predictions where user_id = ?'):
            return Result([{'n':self.db.predictions.count_documents({'user_id':a[0]})}])
        if low.startswith('select * from predictions where user_id = ?'):
            filt={'user_id':a[0]}; ix=1
            if 'and is_healthy = 1' in low: filt['is_healthy']=1
            elif 'and is_healthy = 0' in low: filt['is_healthy']=0
            total=self.db.predictions.count_documents(filt)
            if 'limit ? offset ?' in low:
                per,off=a[-2:]; docs=list(self.db.predictions.find(filt,{'_id':0}).sort('id',-1).skip(off).limit(per))
                # caller separately asks COUNT(*) for this query shape
                return Result(docs)
            return Result(list(self.db.predictions.find(filt,{'_id':0}).sort('id',-1)))
        if low.startswith('insert into users'):
            doc={'id':self._id('users'),'name':a[0],'email':a[1],'password_hash':a[2],'verified':(a[3] if len(a)>3 else 1),'created_at':datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),'last_login_at':None,'failed_attempts':0,'locked_until':None}
            self.db.users.insert_one(doc); self.lastrowid=doc['id']; return Result(lastrowid=doc['id'])
        if low.startswith('insert into predictions'):
            keys=['user_id','image_file','label','crop','condition','is_healthy','confidence','top3_json','latency_ms','note']; doc=dict(zip(keys,a)); doc.update(id=self._id('predictions'),created_at=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')); self.db.predictions.insert_one(doc); self.lastrowid=doc['id']; return Result(lastrowid=doc['id'])
        if low.startswith('insert into password_resets'):
            doc={'id':self._id('password_resets'),'user_id':a[0],'token_hash':a[1],'expires_at':a[2],'used':0}; self.db.password_resets.insert_one(doc); return Result(lastrowid=doc['id'])
        if low.startswith('insert into otps'):
            doc={'id':self._id('otps'),'user_id':a[0],'purpose':a[1],'code_hash':a[2],'expires_at':a[3],'created_at':a[4],'attempts':0,'used':0}; self.db.otps.insert_one(doc); return Result(lastrowid=doc['id'])
        if low.startswith('insert into audit_log'):
            self.db.audit_log.insert_one({'user_id':a[0],'event':a[1],'ip':a[2],'created_at':datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}); return Result()
        if low.startswith('update users set failed_attempts = ?, locked_until = ? where id'):
            self.db.users.update_one({'id':a[2]},{'$set':{'failed_attempts':a[0],'locked_until':a[1]}}); return Result()
        if low.startswith('update users set failed_attempts = 0, locked_until = null, last_login_at'):
            self.db.users.update_one({'id':a[1]},{'$set':{'failed_attempts':0,'locked_until':None,'last_login_at':a[0]}}); return Result()
        if low.startswith('update users set password_hash = ?, failed_attempts = 0'):
            self.db.users.update_one({'id':a[1]},{'$set':{'password_hash':a[0],'failed_attempts':0,'locked_until':None}}); return Result()
        if low.startswith('update users set password_hash = ? where id'):
            self.db.users.update_one({'id':a[1]},{'$set':{'password_hash':a[0]}}); return Result()
        if low.startswith('update users set verified = 1 where id'):
            self.db.users.update_one({'id':a[0]},{'$set':{'verified':1}}); return Result()
        if low.startswith('update users set name = ?, password_hash = ? where id'):
            self.db.users.update_one({'id':a[2]},{'$set':{'name':a[0],'password_hash':a[1]}}); return Result()
        if low.startswith('update otps set used = 1 where user_id'):
            self.db.otps.update_many({'user_id':a[0],'purpose':a[1]},{'$set':{'used':1}}); return Result()
        if low.startswith('update otps set used = 1 where id'):
            self.db.otps.update_one({'id':a[0]},{'$set':{'used':1}}); return Result()
        if low.startswith('update otps set attempts = attempts + 1 where id'):
            self.db.otps.update_one({'id':a[0]},{'$inc':{'attempts':1}}); return Result()
        if low.startswith('update users set name = ? where id'):
            self.db.users.update_one({'id':a[1]},{'$set':{'name':a[0]}}); return Result()
        if low.startswith('update password_resets set used = 1 where user_id'):
            self.db.password_resets.update_many({'user_id':a[0]},{'$set':{'used':1}}); return Result()
        if low.startswith('update users set password_hash'):
            self.db.users.update_one({'id':a[-1]},{'$set':{'password_hash':a[0]}}); return Result()
        if low.startswith('delete from predictions where id'):
            self.db.predictions.delete_one({'id':a[0]}); return Result()
        if low.startswith('delete from users where id'):
            uid=a[0]; self.db.users.delete_one({'id':uid}); self.db.predictions.delete_many({'user_id':uid}); self.db.password_resets.delete_many({'user_id':uid}); self.db.otps.delete_many({'user_id':uid}); return Result()
        raise RuntimeError('MongoDB adapter does not support query: '+s)

def get_db():
    if 'db' not in g:
        # Unit tests may request an explicit temporary SQLite database.
        if current_app.config.get('TESTING') and current_app.config.get('DATABASE'):
            conn=sqlite3.connect(current_app.config['DATABASE']); conn.row_factory=sqlite3.Row; conn.execute('PRAGMA foreign_keys = ON'); g.db=conn
        else:
            from pymongo import MongoClient
            client=MongoClient(current_app.config['MONGO_URI'], serverSelectionTimeoutMS=3000)
            client.admin.command('ping')
            g.mongo_client=client; g.db=MongoDB(client[current_app.config['MONGO_DB']])
    return g.db

def close_db(_=None):
    db=g.pop('db',None)
    if db is not None: db.close()
    client=g.pop('mongo_client',None)
    if client is not None: client.close()

def init_db():
    if current_app.config.get('TESTING') and current_app.config.get('DATABASE'):
        db=get_db(); db.executescript('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT NOT NULL,email TEXT NOT NULL UNIQUE,password_hash TEXT NOT NULL,created_at TEXT NOT NULL DEFAULT (datetime('now')),last_login_at TEXT,failed_attempts INTEGER NOT NULL DEFAULT 0,locked_until TEXT,verified INTEGER NOT NULL DEFAULT 1); CREATE TABLE IF NOT EXISTS otps (id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER NOT NULL,purpose TEXT NOT NULL,code_hash TEXT NOT NULL,expires_at TEXT NOT NULL,created_at TEXT NOT NULL,attempts INTEGER NOT NULL DEFAULT 0,used INTEGER NOT NULL DEFAULT 0); CREATE TABLE IF NOT EXISTS password_resets (id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER NOT NULL,token_hash TEXT NOT NULL UNIQUE,expires_at TEXT NOT NULL,used INTEGER NOT NULL DEFAULT 0); CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER NOT NULL,image_file TEXT NOT NULL,label TEXT NOT NULL,crop TEXT NOT NULL,condition TEXT NOT NULL,is_healthy INTEGER NOT NULL,confidence REAL NOT NULL,top3_json TEXT NOT NULL,latency_ms REAL,note TEXT,created_at TEXT NOT NULL DEFAULT (datetime('now'))); CREATE TABLE IF NOT EXISTS audit_log (id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER,event TEXT NOT NULL,ip TEXT,created_at TEXT NOT NULL DEFAULT (datetime('now')));'''); db.commit(); return
    from pymongo import MongoClient, ASCENDING, DESCENDING
    client=MongoClient(current_app.config['MONGO_URI'],serverSelectionTimeoutMS=3000); client.admin.command('ping'); db=client[current_app.config['MONGO_DB']]
    db.users.create_index([('email',ASCENDING)],unique=True,name='email_unique')
    db.predictions.create_index([('user_id',ASCENDING),('id',DESCENDING)],name='pred_user_id')
    db.password_resets.create_index([('token_hash',ASCENDING)],unique=True,name='reset_token_unique')
    db.password_resets.create_index([('user_id',ASCENDING),('used',ASCENDING)],name='reset_user_used')
    db.otps.create_index([('user_id',ASCENDING),('purpose',ASCENDING),('id',DESCENDING)],name='otp_user_purpose')
    db.otps.create_index([('expires_at',ASCENDING)],name='otp_expiry')
    client.close()

def init_app(app):
    app.teardown_appcontext(close_db)
    with app.app_context(): init_db()

def audit(event: str, user_id=None, ip=None):
    get_db().execute('INSERT INTO audit_log (user_id, event, ip) VALUES (?, ?, ?)',(user_id,event,ip))
