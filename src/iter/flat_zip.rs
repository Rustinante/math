pub trait IntoFlatZipIter<I> {
    fn flat_zip(self, other: I) -> FlatZipIter<I>;
}

pub struct FlatZipIter<I> {
    iters: Vec<I>,
}

impl<I: Iterator> IntoFlatZipIter<I> for I {
    fn flat_zip(self, other: I) -> FlatZipIter<I> {
        FlatZipIter {
            iters: vec![self, other],
        }
    }
}

impl<I: Iterator> IntoFlatZipIter<I> for FlatZipIter<I> {
    fn flat_zip(mut self, other: I) -> FlatZipIter<I> {
        self.iters.push(other);
        FlatZipIter {
            iters: self.iters,
        }
    }
}

impl<I: Iterator> Iterator for FlatZipIter<I> {
    type Item = Vec<<I as Iterator>::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        match self
            .iters
            .iter_mut()
            .map(|i| {
                i.next().map_or_else(|| Err("None encountered"), |x| Ok(x))
            })
            .collect::<Result<Self::Item, &str>>()
        {
            Err(_) => None,
            Ok(v) => Some(v),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::iter::flat_zip::IntoFlatZipIter;

    #[test]
    fn test_flat_zip() {
        let arr1 = vec![1, 2, 3];
        let arr2 = vec![4, 5, 6];
        let arr3 = vec![7, 8, 9];
        let expected_1 =
            vec![vec![&1, &4, &7], vec![&2, &5, &8], vec![&3, &6, &9]];
        for (i1, i2) in arr1
            .iter()
            .flat_zip(arr2.iter())
            .flat_zip(arr3.iter())
            .zip(expected_1.into_iter())
        {
            assert_eq!(i1, i2);
        }

        let expected_2 =
            vec![vec![&1, &4, &7, &10], vec![&2, &5, &8, &11], vec![
                &3, &6, &9, &12,
            ]];
        let arr4 = vec![10, 11, 12, 13];
        for (i1, i2) in arr1
            .iter()
            .flat_zip(arr2.iter())
            .flat_zip(arr3.iter())
            .flat_zip(arr4.iter())
            .zip(expected_2.into_iter())
        {
            assert_eq!(i1, i2);
        }
    }
}
