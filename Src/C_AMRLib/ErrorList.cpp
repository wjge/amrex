//BL_COPYRIGHT_NOTICE

//
// $Id: ErrorList.cpp,v 1.8 1999-05-10 18:54:08 car Exp $
//

#include <ErrorList.H>

const ErrorRec*
ErrorList::operator[] (int k) const
{
    BL_ASSERT(k < length());

    ListIterator<ErrorRec> li(lst);
    
    for ( ; li && k > 0; ++li, --k)
        ;
    return &lst[li];
}

static const char* err_name[] = { "Richardson", "Special" };

ostream&
operator << (ostream&         os,
             const ErrorList& elst)
{
    for (ListIterator<ErrorRec> li(elst.lst); li; ++li)
    {
        os << li().name()
           << ' '
           << li().nGrow()
           << ' '
           << err_name[li().errType()]
           << '\n';
    }
    return os;
}
