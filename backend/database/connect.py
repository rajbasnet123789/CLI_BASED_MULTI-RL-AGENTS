import os

import psycopg2

""" create tables in the PostgreSQL database"""
def create_tables(conn):
    commands = (
        # """
        # CREATE TABLE users_(
        #     id SERIAL PRIMARY KEY,
        #     email VARCHAR(255) NOT NULL,
        #     hash_password VARCHAR(255) NOT NULL
        # )
        # """,
        # """
        # CREATE TABLE tokens_(
        #     id SERIAL PRIMARY KEY,
        #     token VARCHAR(255) NOT NULL,
        #     refresh_token VARCHAR(255) NOT NULL,
        #     user_id INTEGER NOT NULL,
        #     FOREIGN KEY (user_id)
        #         REFERENCES users_ (id)
        #         ON UPDATE CASCADE ON DELETE CASCADE
        #     )
        # """,
        # """
        # CREATE TABLE api_(
        #             id SERIAL PRIMARY KEY,
        #             user_id INTEGER NOT NULL,
        #             api_type VARCHAR(255) NOT NULL,
        #             Models VARCHAR(255) NOT NULL,
        #             FOREIGN KEY (user_id)
        #             REFERENCES users_ (id)
        #             ON UPDATE CASCADE ON DELETE CASCADE
        #     )
        # """,

        # """
        # INSERT INTO users_ (id,email, hash_password)
        # VALUES ('1', 'user@example.com', 'hashed_password')
        # """,
        """
        SELECT * FROM users_
        """
        ,
    )   
    cur = conn.cursor()
    for command in commands:
        cur.execute(command)
        email = cur.fetchone()[1]
        print(f"Email: {email}")
             # close communication with the PostgreSQL database server
    cur.close()
             # commit the changes
    conn.commit()
    

        

def main():
    conn = psycopg2.connect(
        host=os.environ.get("AIVEN_PGHOST", ""),
        port=int(os.environ.get("AIVEN_PGPORT", "5432")),
        database=os.environ.get("AIVEN_PGDATABASE", "defaultdb"),
        user=os.environ.get("AIVEN_PGUSER", ""),
        password=os.environ.get("AIVEN_PGPASSWORD", ""),
        sslmode="require"
    )
    try:
        create_tables(conn)
        print("Tables created successfully in PostgreSQL")
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


    
    

if __name__ == "__main__":
    main()