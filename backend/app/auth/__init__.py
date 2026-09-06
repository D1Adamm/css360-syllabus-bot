"""Authentication and authorization.

Who is asking (`dependencies.current_principal`), what they may do
(`principal.Principal`), and the primitives underneath: opaque tokens,
password hashing, classroom codes, cookies, and rate limiting. The tables
these read and write live in the `db_*` repositories beside the rest.
"""
