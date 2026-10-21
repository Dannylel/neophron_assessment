"""
Task 3 -- A few quick assertions proving permission enforcement works.

This is deliberately light (2-3 checks), not a full test suite -- the
point is the hard requirement they prove, not test-suite completeness.
"""

from app.retrieval import retrieve
from app.users import TEST_USERS


queries=['in Q3 2026 financial summary for Maple Ridge Apartments, what were the operating expens',
         'according to company holiday policy; when is the office  closed ',
         'whats the total revenue for Oak hill plaza 2026 ']

finance={'doc1.txt','doc4.txt','doc6.txt'}
propteam={'doc3.txt','doc6.txt'}
PUBLIC={'doc2.txt','doc5.txt'}

def test_different_users_get_differently_scoped_results():
    # user_a (finance) and user_b (propteam) ask the same query;
    # assert their retrieved chunk sets differ correctly.
    # TEST_USERS['user_a']
    query="in Oak Hill Plaza, when has the Tenant elected to renew for a 12-month term at the existing rate"
    
    docs_a = {
        c.doc_id
        for c in retrieve(query, TEST_USERS["user_a"], top_k=10)
    }
    docs_b = {
        c.doc_id
        for c in retrieve(query, TEST_USERS["user_b"], top_k=10)
        }
    assert docs_a <= finance | PUBLIC              # user_a only sees finance-visible or public docs
    assert docs_b <= propteam | PUBLIC             # user_b only sees propteam-visible or public docs
    assert "doc3.txt" not in docs_a                # propteam-only never leaks to finance
    assert not docs_b & {"doc1.txt", "doc4.txt"}   # finance-only never leaks to propteam
    assert docs_a != docs_b                        # same query, differently scoped results
    
         
    # raise NotImplementedError


def test_all_only_user_never_sees_restricted_chunks():
    # user_c (groups: ["all"]) must never receive a finance- or
    # propteam-only chunk.
    for q in queries: 
            docs_c = {
                c.doc_id
                for c in retrieve(q, TEST_USERS["user_c"], top_k=10)
                }
            assert docs_c <= PUBLIC
            assert docs_c
    
    # raise NotImplementedError
